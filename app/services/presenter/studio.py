"""Glue for the "Presenter video" page: plan -> package -> Kaggle -> assemble.

Long steps run in a background thread and report through
``<job>/status.json``, so closing or reloading the browser page does not stop
a render that takes hours; the page just reads the status again.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import threading
import time
import zipfile

from loguru import logger

from app.models.schema import VideoParams
from app.services import long_script
from app.services.presenter import assemble, planner, profiles
from app.services.presenter import package as job_package
from app.services.presenter.kaggle_agent import KaggleAgent

_threads: dict[str, threading.Thread] = {}


# --------------------------------------------------------------------------- status
def status_path(job_id: str) -> str:
    return os.path.join(job_package.job_dir(job_id), "status.json")


def read_status(job_id: str) -> dict:
    try:
        with open(status_path(job_id), encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return {"state": "new", "log": []}


def set_status(job_id: str, state: str | None = None, message: str = "", **fields) -> None:
    status = read_status(job_id)
    if state:
        status["state"] = state
    if message:
        status.setdefault("log", []).append(f"{time.strftime('%H:%M')} {message}")
        status["log"] = status["log"][-60:]
    status.update(fields, updated=time.time())
    os.makedirs(job_package.job_dir(job_id), exist_ok=True)
    tmp = status_path(job_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(status, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, status_path(job_id))


def is_busy(job_id: str) -> bool:
    thread = _threads.get(job_id)
    return bool(thread and thread.is_alive())


def _background(job_id: str, target, *args) -> None:
    if is_busy(job_id):
        raise RuntimeError("this job is already running")

    def run():
        try:
            target(*args)
        except Exception as exc:
            logger.exception(f"presenter job {job_id} failed")
            set_status(job_id, "error", f"ERROR: {exc}", error=str(exc))

    thread = threading.Thread(target=run, name=f"presenter-{job_id}", daemon=True)
    _threads[job_id] = thread
    thread.start()


# --------------------------------------------------------------------------- planning
def write_script(topic: str, language: str, minutes: float, voice_rate: float = 1.0) -> str:
    return long_script.generate_long_script(topic, language, minutes, voice_rate=voice_rate, with_terms=False).script


def plan_video(topic: str, script: str, language: str, aspect: str, presenter_name: str,
               places: list[dict] | None = None, generate=None) -> dict:
    """Split the script into shots and save ``plan.json``. ``places``: [{name, path, screen}]."""
    presenter = profiles.load_presenter(presenter_name)
    places = [p for p in (places or []) if p.get("name") and os.path.isfile(p.get("path", ""))]
    shots = planner.plan_shots(script, topic, language, presenter.voice_rate,
                               places=[p["name"] for p in places], generate=generate)
    job_id = job_package.new_job_id(topic)
    plan = {"job_id": job_id, "topic": topic, "language": language, "aspect": aspect,
            "presenter": presenter.name, "places": places, "shots": shots, "created": time.time()}
    job_package.save_plan(job_id, plan)
    set_status(job_id, "planned", f"{len(shots)} shots planned")
    return plan


TEST_SHOTS = {
    "ar": [("TALK", "أهلاً وسهلاً! هذا اختبار قصير للمقدّمة الافتراضية.", "a bright modern office",
            "smiles and talks to the camera with a small hand gesture", "medium shot, eye level"),
           ("WALK", "وهنا تمشي بهدوء داخل المكتب.", "a bright modern office",
            "walks slowly towards the camera", "tracking shot")],
    "en": [("TALK", "Hello and welcome! This is a short test of your virtual presenter.", "a bright modern office",
            "smiles and talks to the camera with a small hand gesture", "medium shot, eye level"),
           ("WALK", "And here she walks through the office.", "a bright modern office",
            "walks slowly towards the camera", "tracking shot")],
}


def test_job(presenter_name: str, language: str = "ar-SA", aspect: str = "16:9") -> dict:
    """A tiny ~10 second job (one TALK + one WALK shot) to check the whole chain cheaply."""
    presenter = profiles.load_presenter(presenter_name)
    code = "ar" if str(language).lower().startswith("ar") else "en"
    shots = [{"id": f"s{i + 1:02d}", "type": kind, "narration": text, "location": place, "action": action,
              "camera": camera} for i, (kind, text, place, action, camera) in enumerate(TEST_SHOTS[code])]
    job_id = job_package.new_job_id("test-10s")
    plan = {"job_id": job_id, "topic": "10 second test", "language": language, "aspect": aspect,
            "presenter": presenter.name, "places": [], "shots": shots, "created": time.time()}
    job_package.save_plan(job_id, plan)
    set_status(job_id, "planned", "10 second test job")
    return plan


def update_shots(job_id: str, shots: list[dict]) -> dict:
    """Save the user's edits to the shot list (new ids keep the order)."""
    plan = job_package.load_plan(job_id)
    cleaned = []
    for shot in shots:
        narration = str(shot.get("narration") or "").strip()
        if not narration:
            continue
        kind = str(shot.get("type") or "TALK").upper()
        cleaned.append({
            "id": f"s{len(cleaned) + 1:02d}", "type": kind if kind in planner.SHOT_TYPES else "TALK",
            "narration": narration, "location": str(shot.get("location") or ""),
            "action": str(shot.get("action") or ""), "camera": str(shot.get("camera") or ""),
            **({"place": shot["place"]} if shot.get("place") else {}),
        })
    plan["shots"] = cleaned
    job_package.save_plan(job_id, plan)
    # Narration may have changed: record it again when packaging.
    shutil.rmtree(os.path.join(job_package.job_dir(job_id), "package"), ignore_errors=True)
    return plan


def prepare_package(job_id: str, tts=None) -> str:
    plan = job_package.load_plan(job_id)
    presenter = profiles.load_presenter(plan["presenter"])
    if plan.get("kind") == "marketing":
        presenter.voice_name = profiles.voice_for(presenter, plan.get("language", ""))
    places = {p["name"]: p for p in plan.get("places", [])}
    set_status(job_id, "preparing", "recording the narration")
    folder = job_package.build_package(
        job_id, presenter, plan["shots"], {"aspect": plan["aspect"], "language": plan["language"]}, places, tts=tts,
        progress=lambda done, total: set_status(job_id, progress=done / total * 0.1))
    job_package.save_plan(job_id, plan)  # durations were added to the shots
    return folder


def package_zip(job_id: str) -> bytes:
    """The job package as a zip (for running the worker on Colab by hand)."""
    package = os.path.join(job_package.job_dir(job_id), "package")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(package):
            for name in files:
                path = os.path.join(folder, name)
                zf.write(path, os.path.relpath(path, package))
    return buffer.getvalue()


def import_results(job_id: str, data: bytes) -> int:
    """Unpack a results zip (from Colab) into ``<job>/output``; returns the number of shots."""
    output = os.path.join(job_package.job_dir(job_id), "output")
    count = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            name = os.path.basename(member)
            if not name or member.endswith("/"):
                continue
            if name.endswith(".mp4"):
                sub = "shots"
            elif name.endswith(".png") and "/frames/" in f"/{member}":
                sub = "frames"
            elif name.startswith("candidate_"):
                sub = "candidates"
            elif name in ("progress.json", "summary.json", "worker.log"):
                sub = ""
            else:
                continue
            target_dir = os.path.join(output, sub)
            os.makedirs(target_dir, exist_ok=True)
            with zf.open(member) as src, open(os.path.join(target_dir, name), "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += sub == "shots"
    set_status(job_id, "rendered", f"imported {count} shots")
    return count


def rendered_shots(job_id: str) -> list[str]:
    folder = os.path.join(job_package.job_dir(job_id), "output", "shots")
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


# --------------------------------------------------------------------------- running
def default_params(plan: dict, options: dict | None = None) -> VideoParams:
    options = options or {}
    return VideoParams(
        video_subject=plan.get("topic", "") or "presenter video",
        video_aspect=plan.get("aspect", "16:9"),
        video_language=plan.get("language", ""),
        subtitle_enabled=options.get("subtitles", True),
        subtitle_style=options.get("subtitle_style", "boxed"),
        font_name=options.get("font_name") or "Tajawal-Bold.ttf",  # Arabic + Latin
        bgm_type=options.get("bgm_type", "random"),
        bgm_volume=float(options.get("bgm_volume", 0.12)),
        add_logo_watermark=bool(options.get("logo", False)),
        add_intro_outro=bool(options.get("intro_outro", False)),
        n_threads=int(options.get("threads", 2)),
    )


def render_final(job_id: str, options: dict | None = None) -> str:
    plan = job_package.load_plan(job_id)
    if plan.get("kind") == "marketing":  # website ad: composited scenes
        from app.services.marketing import pipeline

        return pipeline.render(job_id, options)
    set_status(job_id, "assembling", "assembling the final video")
    final = assemble.assemble(job_id, default_params(plan, options),
                              progress=lambda p: set_status(job_id, progress=0.9 + p * 0.1))
    set_status(job_id, "done", "the video is ready", final=final, progress=1.0)
    return final


def _has_previous_output(job_id: str) -> bool:
    return os.path.isfile(os.path.join(job_package.job_dir(job_id), "output", "progress.json"))


def _run_on_kaggle(job_id: str, token: str, options: dict, agent=None) -> None:
    if not os.path.isfile(os.path.join(job_package.job_dir(job_id), "package", "job.json")):
        prepare_package(job_id)
    agent = agent or KaggleAgent(token=token, log=lambda m: set_status(job_id, message=m))
    set_status(job_id, "running", "sending the job to Kaggle")
    # Shots finished by an earlier run go up with the job and are not rendered again.
    summary = agent.run_job(job_id, on_status=lambda s: set_status(job_id, kaggle=s),
                            continuing=_has_previous_output(job_id))
    set_status(job_id, "rendered", f"{summary.get('done', 0)}/{summary.get('total', 0)} shots rendered",
               summary=summary)
    render_final(job_id, options)


def start_kaggle_render(job_id: str, token: str, options: dict | None = None, agent=None) -> None:
    set_status(job_id, "queued", "starting", error="")
    _background(job_id, _run_on_kaggle, job_id, token, options or {}, agent)


ACTIVE_STATES = {"queued", "preparing", "running", "rendered", "assembling"}


def was_interrupted(job_id: str) -> bool:
    """The job was working when the program stopped (laptop off, window closed, drive removed)."""
    return read_status(job_id).get("state") in ACTIVE_STATES and not is_busy(job_id)


def has_kaggle_run(job_id: str) -> bool:
    try:
        with open(os.path.join(job_package.job_dir(job_id), "kaggle.json"), encoding="utf-8") as fp:
            return bool(json.load(fp).get("kernel"))
    except (OSError, ValueError):
        return False


def can_continue(job_id: str) -> bool:
    """Continue is offered after a break, and after a local error once a Kaggle run exists
    (its results can still be fetched without running the GPU again)."""
    if is_busy(job_id):
        return False
    return was_interrupted(job_id) or (read_status(job_id).get("state") == "error" and has_kaggle_run(job_id))


def is_creation_job(job_id: str) -> bool:
    try:
        return job_package.load_plan(job_id).get("kind") == "create_presenter"
    except (OSError, ValueError):
        return False


def _resume(job_id: str, token: str, options: dict, agent=None) -> None:
    agent = agent or KaggleAgent(token=token, log=lambda m: set_status(job_id, message=m))
    creation = is_creation_job(job_id)
    set_status(job_id, "running", "continuing: checking Kaggle", error="")
    package_ready = os.path.isfile(os.path.join(job_package.job_dir(job_id), "package", "job.json"))
    if not creation and not package_ready:
        prepare_package(job_id)
    # Continue only fetches: it never starts a new GPU run by itself (max_runs=1 means "the
    # last run only"). Missing shots are reported; rendering them is the user's explicit choice.
    summary = agent.resume(job_id, max_runs=1, on_status=lambda s: set_status(job_id, kaggle=s))
    if creation:
        set_status(job_id, "done", "candidates ready")
        return
    missing = summary.get("remaining") or []
    set_status(job_id, "rendered", f"{summary.get('done', 0)}/{summary.get('total', 0)} shots rendered"
               + (f"; missing: {', '.join(missing)} (press Render to make only these)" if missing else ""),
               summary=summary)
    render_final(job_id, options)


def start_resume(job_id: str, token: str, options: dict | None = None, agent=None) -> None:
    """Continue a job after a break: download finished work from Kaggle, run what is missing, assemble."""
    _background(job_id, _resume, job_id, token, options or {}, agent)


def creation_jobs() -> list[str]:
    return [j for j in job_package.list_jobs() if is_creation_job(j)]


def start_assembly(job_id: str, options: dict | None = None) -> None:
    _background(job_id, render_final, job_id, options or {})


# --------------------------------------------------------------------------- new presenter
def _create_presenter(job_id: str, token: str, description: str, count: int, agent=None) -> None:
    job_package.build_create_presenter_package(job_id, description, count)
    agent = agent or KaggleAgent(token=token, log=lambda m: set_status(job_id, message=m))
    set_status(job_id, "running", "creating presenter photos on Kaggle")
    agent.run_job(job_id, max_runs=1, on_status=lambda s: set_status(job_id, kaggle=s))
    set_status(job_id, "done", "candidates ready")


def start_presenter_creation(description: str, token: str, count: int = 4, agent=None) -> str:
    job_id = job_package.new_job_id("presenter")
    job_package.save_plan(job_id, {"job_id": job_id, "kind": "create_presenter", "description": description})
    set_status(job_id, "queued", "starting")
    _background(job_id, _create_presenter, job_id, token, description, count, agent)
    return job_id


def candidates(job_id: str) -> list[str]:
    folder = os.path.join(job_package.job_dir(job_id), "output", "candidates")
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".png")]
