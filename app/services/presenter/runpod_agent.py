"""Runs presenter jobs on a RunPod Serverless endpoint (paid, faster GPUs).

Same job package and the same worker script as Kaggle (``runpod/handler.py``
wraps ``kaggle/presenter_worker.py``). The flow:

1. zip the job package and send it with ``/run`` (mode ``render``);
2. poll ``/status`` until the run ends (it keeps going if the laptop is off);
3. ask the endpoint for the list of files it kept on its network volume and
   download them in pieces (mode ``list`` / ``fetch``), then delete them there.

Needs an API key and the endpoint id, kept in the git-ignored ``config.toml``
as ``runpod_api_key`` / ``runpod_endpoint_id``. Every call here that reaches
RunPod can cost money; the tests use a fake session only.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import zipfile

from loguru import logger

from app.config import config
from app.services.presenter import package as job_package

API = "https://api.runpod.ai/v2"
MAX_PACKAGE = 7 * 1024 * 1024  # zipped job; base64 of it must stay under RunPod's 10 MB /run limit
PIECE = 4 * 1024 * 1024
# RunPod job states -> the names the rest of the program already understands (Kaggle's).
STATES = {"IN_QUEUE": "queued", "IN_PROGRESS": "running", "COMPLETED": "complete",
          "FAILED": "error", "CANCELLED": "error", "TIMED_OUT": "error"}
FINISHED = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodError(RuntimeError):
    pass


def configured_key() -> str:
    return str(config.app.get("runpod_api_key", "") or os.environ.get("RUNPOD_API_KEY", "")).strip()


def configured_endpoint() -> str:
    return str(config.app.get("runpod_endpoint_id", "") or "").strip()


class RunPodAgent:
    def __init__(self, api_key: str | None = None, endpoint_id: str | None = None, session=None,
                 poll_seconds: int = 20, log=None, sleep=time.sleep, time_budget: float = 3 * 3600):
        self.api_key = api_key if api_key is not None else configured_key()
        self.endpoint_id = endpoint_id if endpoint_id is not None else configured_endpoint()
        self._session = session
        self.poll_seconds = poll_seconds
        self.log = log or (lambda message: logger.info(message))
        self.sleep = sleep
        self.time_budget = time_budget
        self.last_run: dict = {}

    # -- connection -------------------------------------------------------------
    @property
    def session(self):
        if not self.api_key or not self.endpoint_id:
            raise RunPodError("RunPod API key or endpoint id is missing. Add them in the Presenter video settings.")
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float = 120) -> dict:
        url = f"{API}/{self.endpoint_id}/{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last = None
        for attempt in range(4):
            try:
                response = self.session.request(method, url, json=payload, headers=headers, timeout=timeout)
            except Exception as exc:  # network hiccup
                last = exc
                self.sleep(5 * (attempt + 1))
                continue
            if response.status_code in (401, 403):
                raise RunPodError("RunPod rejected the API key.")
            if response.status_code == 404:
                return {"status": "NOT_FOUND"}
            if response.status_code >= 500 or response.status_code == 429:
                last = RunPodError(f"RunPod answered {response.status_code}")
                self.sleep(5 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise RunPodError(f"RunPod answered {response.status_code}: {response.text[:300]}")
            return response.json()
        raise RunPodError(f"RunPod could not be reached: {last}")

    def check(self) -> str:
        """Returns a short health line if the key and endpoint work."""
        health = self._request("GET", "health", timeout=30)
        if health.get("status") == "NOT_FOUND":
            raise RunPodError("RunPod endpoint not found; check the endpoint id.")
        workers = health.get("workers", {})
        return f"endpoint {self.endpoint_id}: {workers.get('idle', 0)} idle / {workers.get('running', 0)} running workers"

    def call(self, payload: dict, timeout: float = 1800) -> dict:
        """A short request (list / fetch / cleanup): wait for its answer."""
        answer = self._request("POST", "runsync", {"input": payload}, timeout=150)
        deadline = time.time() + timeout
        while answer.get("status") not in FINISHED:
            if answer.get("status") == "NOT_FOUND" or time.time() > deadline:
                raise RunPodError(f"RunPod did not answer the {payload.get('mode')} request")
            self.sleep(3)
            answer = self._request("GET", f"status/{answer['id']}")
        output = answer.get("output")
        if answer.get("status") != "COMPLETED" or not isinstance(output, dict):
            raise RunPodError(f"RunPod {payload.get('mode')} request failed: {answer.get('error') or answer}")
        if output.get("error"):
            raise RunPodError(output["error"])
        return output

    # -- steps ----------------------------------------------------------------------
    def zip_package(self, job_id: str) -> str:
        package = os.path.join(job_package.job_dir(job_id), "package")
        if not os.path.isfile(os.path.join(package, "job.json")):
            raise RunPodError("job package not found; prepare the job first")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder, dirs, files in os.walk(package):
                # Finished shots of an earlier run stay on the endpoint's volume; never re-upload them.
                dirs[:] = [d for d in dirs if not (folder == package and d == "previous")]
                for name in files:
                    path = os.path.join(folder, name)
                    zf.write(path, os.path.relpath(path, package).replace(os.sep, "/"))
        data = buffer.getvalue()
        if len(data) > MAX_PACKAGE:
            raise RunPodError(f"The job package is {len(data) / 1e6:.1f} MB; RunPod accepts up to "
                              f"{MAX_PACKAGE / 1e6:.0f} MB. Use smaller presenter / place photos.")
        return base64.b64encode(data).decode("ascii")

    def submit(self, job_id: str, fresh: bool) -> str:
        payload = {"mode": "render", "job_id": job_id, "package": self.zip_package(job_id), "fresh": fresh,
                   "time_budget": self.time_budget}
        policy = {"executionTimeout": int((self.time_budget + 1800) * 1000)}
        answer = self._request("POST", "run", {"input": payload, "policy": policy})
        request = answer.get("id")
        if not request:
            raise RunPodError(f"RunPod did not accept the job: {answer}")
        self._save_state(job_id, request=request, endpoint=self.endpoint_id)
        self.log(f"job sent to RunPod (request {request})")
        return request

    def wait(self, request: str, on_status=None, max_hours: float = 8) -> tuple[str, str]:
        """Poll until the run ends: (RunPod state or NOT_FOUND once RunPod forgot it, error text)."""
        deadline = time.time() + max_hours * 3600
        last, last_message = "", ""
        while time.time() < deadline:
            try:
                answer = self._request("GET", f"status/{request}")
            except RunPodError as exc:
                if "rejected" in str(exc):
                    raise
                self.log(f"status check failed ({exc}); retrying")
                self.sleep(self.poll_seconds)
                continue
            state = answer.get("status", "")
            if state != last:
                self.log(f"RunPod status: {state}")
                last = state
            if on_status and state in STATES:
                on_status(STATES[state])
            message = answer.get("output") if state == "IN_PROGRESS" else None
            if isinstance(message, str) and message != last_message:
                self.log(message)
                last_message = message
            if state in FINISHED or state == "NOT_FOUND":
                output = answer.get("output")
                error = answer.get("error") or (output.get("error") if isinstance(output, dict) else "") or ""
                if error:
                    self.log(f"RunPod error: {str(error)[:800]}")  # shown right away on the page
                self.last_run = {"state": state, "queue_seconds": (answer.get("delayTime") or 0) / 1000,
                                 "execution_seconds": (answer.get("executionTime") or 0) / 1000,
                                 "gpu": output.get("gpu", "") if isinstance(output, dict) else ""}
                return state, str(error)
            self.sleep(self.poll_seconds)
        return "TIMEOUT", ""

    def cancel(self, job_id: str) -> bool:
        """Stop the job's running request (stops the GPU billing); finished shots stay on the volume."""
        request = self._state(job_id).get("request")
        if not request:
            return False
        answer = self._request("POST", f"cancel/{request}")
        self.log(f"RunPod request {request}: cancel -> {answer.get('status', 'sent')}")
        return True

    def download(self, job_id: str) -> dict:
        """Copy what the endpoint kept for this job into ``<job>/output``; returns summary.json (or {})."""
        started = time.time()
        listing = self.call({"mode": "list", "job_id": job_id})
        output = os.path.join(job_package.job_dir(job_id), "output")
        for sub in ("shots", "frames", "candidates"):
            os.makedirs(os.path.join(output, sub), exist_ok=True)
        for item in listing.get("files", []):
            relative = str(item["path"])
            if relative.startswith(("/", "\\")) or ".." in relative.replace("\\", "/").split("/"):
                continue
            target = os.path.join(output, *relative.split("/"))
            size = int(item.get("size", 0))
            if os.path.isfile(target) and os.path.getsize(target) == size and relative.startswith(("shots/", "frames/", "candidates/")):
                continue  # already here
            partial = target + ".part"
            with open(partial, "wb") as fp:
                offset = 0
                while offset < size:
                    piece = self.call({"mode": "fetch", "job_id": job_id, "path": relative,
                                       "offset": offset, "length": PIECE})
                    data = base64.b64decode(piece["data"])
                    if not data:
                        raise RunPodError(f"empty piece while downloading {relative}")
                    fp.write(data)
                    offset += len(data)
            os.replace(partial, target)
        self.log(f"downloaded {len(listing.get('files', []))} files from RunPod")
        timings = dict(self.last_run)
        timings["download_seconds"] = round(time.time() - started, 1)
        self._save_state(job_id, timings=timings)  # for benchmark.json
        return listing.get("summary") or {}

    def cleanup(self, job_id: str) -> None:
        try:
            self.call({"mode": "cleanup", "job_id": job_id})
        except RunPodError as exc:  # a leftover folder only costs volume space
            self.log(f"could not delete the job files on RunPod ({exc})")

    def run_job(self, job_id: str, max_runs: int = 2, on_status=None, continuing: bool = False) -> dict:
        """Send, wait, download; a second run only continues shots the first could not finish."""
        summary: dict = {}
        for attempt in range(1, max_runs + 1):
            request = self.submit(job_id, fresh=not (continuing or attempt > 1))
            summary, finished = self._collect(job_id, request, on_status)
            if finished:
                return summary
            if attempt < max_runs:
                self.log("starting another run for the missing shots")
        return summary

    def has_run(self, job_id: str) -> bool:
        return bool(self._state(job_id).get("request"))

    def resume(self, job_id: str, max_runs: int = 1, on_status=None) -> dict:
        """After a break: wait for the last run if it still goes, download what it made."""
        request = self._state(job_id).get("request")
        if not request:
            self.log("this job was never sent to RunPod; sending it now")
            return self.run_job(job_id, max_runs=max_runs, on_status=on_status)
        self.log(f"checking the last RunPod run ({request})")
        summary, finished = self._collect(job_id, request, on_status)
        if finished or max_runs <= 1:
            return summary
        return self.run_job(job_id, max_runs=max_runs - 1, on_status=on_status, continuing=True)

    def _collect(self, job_id: str, request: str, on_status=None) -> tuple[dict, bool]:
        state, error = self.wait(request, on_status=on_status)
        if state == "TIMEOUT":
            raise RunPodError("Stopped waiting for RunPod; press Continue later to fetch the result.")
        try:
            summary = self.download(job_id)
        except RunPodError as exc:
            if state != "COMPLETED" or error:  # the run's own error explains more than the fetch
                if on_status:
                    on_status("error")
                raise RunPodError(f"The RunPod run ended ({state}): {error or exc}") from exc
            raise
        done = bool(summary.get("complete")) or (self._is_presenter_job(job_id) and state in ("COMPLETED", "NOT_FOUND"))
        if done:
            self.log("all shots are ready")
            self.cleanup(job_id)
            return summary, True
        if (state != "COMPLETED" or error) and not summary.get("done"):
            if on_status:
                on_status("error")
            raise RunPodError(f"The RunPod run ended ({state}) before making any shot. {error}".strip())
        self.log(f"{summary.get('done', 0)}/{summary.get('total', '?')} shots ready")
        return summary, False

    # -- helpers --------------------------------------------------------------------
    def _is_presenter_job(self, job_id: str) -> bool:
        with open(os.path.join(job_package.job_dir(job_id), "package", "job.json"), encoding="utf-8") as fp:
            return json.load(fp).get("kind") == "create_presenter"

    def _state(self, job_id: str) -> dict:
        try:
            with open(os.path.join(job_package.job_dir(job_id), "runpod.json"), encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, ValueError):
            return {}

    def _save_state(self, job_id: str, **fields) -> None:
        state = self._state(job_id)
        state.update(fields)
        with open(os.path.join(job_package.job_dir(job_id), "runpod.json"), "w", encoding="utf-8") as fp:
            json.dump(state, fp, ensure_ascii=False, indent=2)
