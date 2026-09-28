"""RunPod Serverless entry point for the presenter worker.

The GPU work is exactly ``kaggle/presenter_worker.py`` (the same script Kaggle
runs); this file only moves files in and out. Everything a job makes stays on
the endpoint's network volume (``/runpod-volume``) until the laptop has
downloaded it, so a finished run is never lost when the laptop was off, and
the ~60 GB of models download once instead of on every start.

Requests (``input``):

* ``{"mode": "render", "job_id", "package": <base64 zip>, "fresh": bool, "time_budget": seconds}``
  run the job; ``fresh`` drops shots left from an earlier run of the same id.
* ``{"mode": "list", "job_id"}`` -> summary, progress, file list, last log lines.
* ``{"mode": "fetch", "job_id", "path", "offset", "length"}`` -> one piece of a file (base64).
* ``{"mode": "cleanup", "job_id"}`` -> delete the job's folder once the laptop has everything.
* ``{"mode": "tts", "engine": "silma", "items": [...]}`` -> short speech clips (see ``tts``).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import sys
import threading
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VOLUME = os.environ.get("MPT_VOLUME", "/runpod-volume")
COMFY_DIR = os.environ.get("MPT_COMFY_DIR", "/comfy")
MAX_PIECE = 4 * 1024 * 1024  # bytes per fetch answer (base64 must stay well under RunPod's limits)
DEFAULT_BUDGET = 3 * 3600
OUTPUT_DIRS = ("shots", "frames", "candidates")


class HandlerError(RuntimeError):
    pass


def _job_id(value) -> str:
    job_id = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", job_id) or job_id.startswith("."):
        raise HandlerError(f"bad job_id: {job_id!r}")
    return job_id


def job_root(job_id: str) -> str:
    if not os.path.isdir(VOLUME):
        raise HandlerError("No network volume is attached to this endpoint (/runpod-volume). "
                           "Attach one in the endpoint settings; results are kept there.")
    return os.path.join(VOLUME, "jobs", _job_id(job_id))


def _inside(root: str, relative: str) -> str:
    path = os.path.realpath(os.path.join(root, relative))
    if not path.startswith(os.path.realpath(root) + os.sep):
        raise HandlerError(f"path outside the job: {relative!r}")
    return path


def _unpack(data: str, target: str) -> None:
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(data))) as zf:
        for member in zf.namelist():
            _inside(target, member)  # refuse "../" entries
        zf.extractall(target)


def _tail(path: str, lines: int = 30) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            return fp.read().splitlines()[-lines:]
    except OSError:
        return []


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return {}


def render(job: dict, progress=None) -> dict:
    job_id = _job_id(job["input"].get("job_id"))
    root = job_root(job_id)
    out = os.path.join(root, "output")
    if job["input"].get("fresh"):
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    _unpack(job["input"]["package"], os.path.join(root, "package"))

    store = os.path.join(VOLUME, "models")
    os.makedirs(store, exist_ok=True)
    os.environ["MPT_MODEL_STORE"] = store
    import presenter_worker

    budget = float(job["input"].get("time_budget") or DEFAULT_BUDGET)
    argv = ["--job", os.path.join(root, "package"), "--out", out, "--work", os.path.join("/tmp", "work", job_id),
            "--comfy", COMFY_DIR, "--cache", store, "--skip-setup", "--time-budget", str(budget)]
    done = threading.Event()
    result: dict = {}

    def work():
        try:
            result["code"] = presenter_worker.main(argv)
        except BaseException as exc:  # SystemExit from the worker carries its message
            result["error"] = str(exc) or exc.__class__.__name__
        finally:
            done.set()

    threading.Thread(target=work, daemon=True).start()
    log_path = os.path.join(out, "worker.log")
    while not done.wait(20):
        if progress:
            last = _tail(log_path, 1)
            if last:
                progress(job, last[0][-300:])
    listing = list_job(job_id)
    listing["gpu"] = gpu_name()
    if result.get("error") and not listing["summary"].get("done"):
        raise HandlerError(f"{result['error']}\n" + "\n".join(listing["log"][-15:]))
    listing["error"] = result.get("error", "")
    return listing


def gpu_name() -> str:
    try:
        import subprocess

        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def list_job(job_id: str) -> dict:
    out = os.path.join(job_root(job_id), "output")
    files = []
    for sub in OUTPUT_DIRS:
        folder = os.path.join(out, sub)
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                path = os.path.join(folder, name)
                if os.path.isfile(path) and not name.endswith((".tmp", ".tmp.mp4")):
                    files.append({"path": f"{sub}/{name}", "size": os.path.getsize(path)})
    for name in ("summary.json", "progress.json", "worker.log"):
        if os.path.isfile(os.path.join(out, name)):
            files.append({"path": name, "size": os.path.getsize(os.path.join(out, name))})
    return {"job_id": job_id, "exists": os.path.isdir(out), "summary": _read_json(os.path.join(out, "summary.json")),
            "files": files, "log": _tail(os.path.join(out, "worker.log"))}


def fetch(inp: dict) -> dict:
    out = os.path.join(job_root(inp.get("job_id")), "output")
    path = _inside(out, str(inp.get("path", "")))
    offset = max(0, int(inp.get("offset", 0)))
    length = min(MAX_PIECE, max(1, int(inp.get("length", MAX_PIECE))))
    with open(path, "rb") as fp:
        fp.seek(offset)
        data = fp.read(length)
    return {"path": inp["path"], "offset": offset, "size": os.path.getsize(path),
            "data": base64.b64encode(data).decode("ascii")}


def cleanup(job_id: str) -> dict:
    shutil.rmtree(job_root(job_id), ignore_errors=True)
    shutil.rmtree(os.path.join("/tmp", "work", _job_id(job_id)), ignore_errors=True)
    return {"job_id": job_id, "deleted": True}


SILMA_PYTHON = os.environ.get("MPT_SILMA_PYTHON", "/opt/silma/bin/python")
TTS_ENGINES = {"silma": os.path.join(HERE, "tts_silma.py")}
MAX_TTS_ITEMS = 12


def tts(inp: dict) -> dict:
    """Short speech clips (Voice Lab, narration): {"engine", "items": [{"id", "text", "ref_wav" (base64 WAV),
    "ref_text", "speed", "seed"}]} -> {"items": [{"id", "wav" (base64), "seconds", ...}], timings}."""
    import subprocess
    import tempfile

    engine = str(inp.get("engine", ""))
    if engine not in TTS_ENGINES:
        raise HandlerError(f"unknown tts engine {engine!r}")
    items = list(inp.get("items") or [])[:MAX_TTS_ITEMS]
    work = tempfile.mkdtemp(prefix="tts_")
    request = []
    for index, item in enumerate(items):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", str(item.get("id", ""))):
            raise HandlerError(f"bad item id: {item.get('id')!r}")
        entry = {k: item.get(k) for k in ("id", "text", "ref_text", "speed", "seed")}
        if item.get("ref_wav"):
            ref = os.path.join(work, f"ref_{index}.wav")
            with open(ref, "wb") as fp:
                fp.write(base64.b64decode(item["ref_wav"]))
            entry["ref_wav"] = ref
        request.append(entry)
    with open(os.path.join(work, "request.json"), "w", encoding="utf-8") as fp:
        json.dump({"items": request}, fp, ensure_ascii=False)
    env = dict(os.environ)
    cache = os.path.join(VOLUME, "hf") if os.path.isdir(VOLUME) else os.path.join(work, "hf")
    env.update({"HF_HOME": cache, "PYTHONUTF8": "1"})
    out = os.path.join(work, "out")
    run = subprocess.run([SILMA_PYTHON, TTS_ENGINES[engine], os.path.join(work, "request.json"), out],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if run.returncode:
        raise HandlerError(f"{engine} failed: {(run.stderr or run.stdout)[-1500:]}")
    result = _read_json(os.path.join(out, "result.json"))
    for item in result.get("items", []):
        with open(os.path.join(out, item.pop("file")), "rb") as fp:
            item["wav"] = base64.b64encode(fp.read()).decode("ascii")
    result["gpu"] = gpu_name()
    shutil.rmtree(work, ignore_errors=True)
    return result


def handler(job: dict, progress=None) -> dict:
    inp = job.get("input") or {}
    mode = inp.get("mode", "render")
    try:
        if mode == "render":
            return render(job, progress)
        if mode == "list":
            return list_job(_job_id(inp.get("job_id")))
        if mode == "fetch":
            return fetch(inp)
        if mode == "cleanup":
            return cleanup(_job_id(inp.get("job_id")))
        if mode == "tts":
            return tts(inp)
        raise HandlerError(f"unknown mode {mode!r}")
    except (HandlerError, OSError, ValueError, KeyError) as exc:
        return {"error": str(exc)}


if __name__ == "__main__":  # pragma: no cover - runs inside the RunPod image
    import runpod

    def _progress(job, message):
        runpod.serverless.progress_update(job, message)

    runpod.serverless.start({"handler": lambda job: handler(job, _progress)})
