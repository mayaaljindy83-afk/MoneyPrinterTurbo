"""Runs presenter jobs on Kaggle's free GPU through the official Kaggle API.

One button on the laptop does everything:

1. zip the job package and upload it as a *private* Kaggle dataset;
2. push ``kaggle/presenter_worker.py`` as a private script with GPU + internet on;
3. poll until the run ends, download its output (the finished shots);
4. if shots are left (session limit, an error), put the finished shots back
   into the dataset (``previous/``) and run again, until all are done.

Only a Kaggle API token is needed (kaggle.com -> Settings -> API -> Generate
New Token). It is kept in the git-ignored ``config.toml`` as
``kaggle_api_token``. Nothing here costs money: Kaggle gives ~30 GPU hours a
week for free and a run that goes over simply stops.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile

from loguru import logger

from app.config import config
from app.services.presenter import package as job_package
from app.utils import utils

WORKER_SCRIPT = os.path.join(utils.root_dir(), "kaggle", "presenter_worker.py")
MACHINE = "NvidiaTeslaT4"
FINISHED = {"complete", "error", "cancelacknowledged", "cancel_acknowledged"}
RUNNING = {"queued", "running", "new_script", "newscript", "cancelrequested", "cancel_requested"}


class KaggleError(RuntimeError):
    pass


def configured_token() -> str:
    return str(config.app.get("kaggle_api_token", "") or os.environ.get("KAGGLE_API_TOKEN", "")).strip()


def _slug(text: str, limit: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-")


def status_name(response) -> str:
    """Normalize ``kernels_status`` answers (enum, string or object) to e.g. ``running``."""
    status = getattr(response, "status", response)
    name = getattr(status, "name", None) or str(status)
    return name.split(".")[-1].strip().lower()


class KaggleAgent:
    def __init__(self, token: str | None = None, api=None, poll_seconds: int = 60, log=None, sleep=time.sleep):
        self.token = token if token is not None else configured_token()
        self._api = api
        self.poll_seconds = poll_seconds
        self.log = log or (lambda message: logger.info(message))
        self.sleep = sleep

    # -- connection -------------------------------------------------------------
    @property
    def api(self):
        if self._api is None:
            if not self.token:
                raise KaggleError("Kaggle API token is missing. Add it in the Presenter video settings.")
            os.environ["KAGGLE_API_TOKEN"] = self.token
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError as exc:  # pragma: no cover - dependency is in requirements
                raise KaggleError("The 'kaggle' package is not installed. Run install.bat again.") from exc
            api = KaggleApi()
            try:
                api.authenticate()
            except (Exception, SystemExit) as exc:
                raise KaggleError(f"Kaggle rejected the token: {exc}") from exc
            self._api = api
        return self._api

    @property
    def username(self) -> str:
        name = self.api.get_config_value("username")
        if not name:
            raise KaggleError("Could not read the Kaggle user name from the token.")
        return name

    def check(self) -> str:
        """Returns the user name if the token works."""
        return self.username

    # -- references ---------------------------------------------------------------
    def dataset_ref(self, job_id: str) -> str:
        return f"{self.username}/{_slug('mpt-job-' + job_id)}"

    def kernel_ref(self, job_id: str) -> str:
        return f"{self.username}/{_slug('mpt-presenter-' + job_id)}"

    # -- steps ----------------------------------------------------------------------
    def upload_job(self, job_id: str) -> str:
        """Upload (or update) the job package as a private dataset; returns its ref."""
        root = job_package.job_dir(job_id)
        package = os.path.join(root, "package")
        if not os.path.isfile(os.path.join(package, "job.json")):
            raise KaggleError("job package not found; prepare the job first")
        upload = os.path.join(root, "upload")
        shutil.rmtree(upload, ignore_errors=True)
        os.makedirs(upload)
        with zipfile.ZipFile(os.path.join(upload, "package.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            for folder, _, files in os.walk(package):
                for name in files:
                    path = os.path.join(folder, name)
                    zf.write(path, os.path.relpath(path, package))
        ref = self.dataset_ref(job_id)
        with open(os.path.join(upload, "dataset-metadata.json"), "w", encoding="utf-8") as fp:
            json.dump({"title": ref.split("/", 1)[1], "id": ref, "licenses": [{"name": "CC0-1.0"}]}, fp)
        state = os.path.join(root, "kaggle.json")
        uploaded = os.path.isfile(state) and self._state(job_id).get("dataset") == ref
        if uploaded:
            self.log(f"updating Kaggle dataset {ref}")
            self._check(self.api.dataset_create_version(upload, "resume", quiet=True, dir_mode="skip"))
        else:
            self.log(f"uploading job to private Kaggle dataset {ref}")
            self._check(self.api.dataset_create_new(upload, public=False, quiet=True, dir_mode="skip"))
            self._save_state(job_id, dataset=ref)
        self._wait_dataset(ref)
        return ref

    def push_worker(self, job_id: str, dataset: str) -> str:
        """Start the worker as a private GPU script with the job dataset attached."""
        root = job_package.job_dir(job_id)
        folder = os.path.join(root, "kernel")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        shutil.copyfile(WORKER_SCRIPT, os.path.join(folder, "presenter_worker.py"))
        ref = self.kernel_ref(job_id)
        metadata = {
            "id": ref, "title": ref.split("/", 1)[1], "code_file": "presenter_worker.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": MACHINE,
            "dataset_sources": [dataset], "kernel_sources": [], "competition_sources": [],
        }
        with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
        self.log(f"starting the worker on Kaggle GPU: https://www.kaggle.com/code/{ref}")
        self._check(self.api.kernels_push(folder))
        self._save_state(job_id, kernel=ref)
        return ref

    def wait(self, kernel: str, on_status=None, max_hours: float = 13) -> str:
        """Poll until the run ends; returns the final status name."""
        deadline = time.time() + max_hours * 3600
        last = ""
        while time.time() < deadline:
            try:
                status = status_name(self.api.kernels_status(kernel))
            except Exception as exc:  # network hiccup: keep waiting
                self.log(f"status check failed ({exc}); retrying")
                status = last or "unknown"
            if status != last:
                self.log(f"Kaggle run status: {status}")
                last = status
            if on_status:
                on_status(status)
            if status in FINISHED:
                return status
            self.sleep(self.poll_seconds)
        return "timeout"

    def download(self, job_id: str, kernel: str) -> dict:
        """Download the run output into ``<job>/output``; returns summary.json (or {})."""
        output = os.path.join(job_package.job_dir(job_id), "output")
        os.makedirs(output, exist_ok=True)
        token = None
        while True:
            result = self.api.kernels_output(kernel, output, force=True, quiet=True, page_token=token)
            token = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            if not token:
                break
        # Kaggle may keep the worker's sub-folders or flatten them; normalise.
        for sub in ("shots", "frames", "candidates"):
            os.makedirs(os.path.join(output, sub), exist_ok=True)
        for name in os.listdir(output):
            path = os.path.join(output, name)
            if name.endswith(".mp4") and os.path.isfile(path):
                shutil.move(path, os.path.join(output, "shots", name))
            elif name.startswith("candidate_") and os.path.isfile(path):
                shutil.move(path, os.path.join(output, "candidates", name))
        summary_path = os.path.join(output, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def run_job(self, job_id: str, max_runs: int = 4, on_status=None, continuing: bool = False) -> dict:
        """Upload, run, download and repeat until every shot is rendered.

        ``continuing``: an earlier run already produced output, so the first
        upload carries its finished shots too.
        """
        summary: dict = {}
        for attempt in range(1, max_runs + 1):
            if attempt > 1 or continuing:
                kept = job_package.stage_previous_output(job_id)
                self.log(f"run {attempt}: {kept} shots already done, continuing with the rest")
            dataset = self.upload_job(job_id)
            kernel = self.push_worker(job_id, dataset)
            summary, finished = self._collect(job_id, kernel, on_status)
            if finished:
                return summary
        return summary

    def resume(self, job_id: str, max_runs: int = 4, on_status=None) -> dict:
        """Pick a job up again after the laptop was off or the program closed.

        The run on Kaggle keeps going without the laptop: wait for it if it
        is still running, download what it made, and start more runs only if
        shots are still missing.
        """
        kernel = self._state(job_id).get("kernel")
        if not kernel:
            self.log("this job was never sent to Kaggle; sending it now")
            return self.run_job(job_id, max_runs=max_runs, on_status=on_status)
        self.log(f"checking the last Kaggle run: https://www.kaggle.com/code/{kernel}")
        summary, finished = self._collect(job_id, kernel, on_status)
        if finished or max_runs <= 1:
            return summary
        return self.run_job(job_id, max_runs=max_runs - 1, on_status=on_status, continuing=True)

    def _collect(self, job_id: str, kernel: str, on_status=None) -> tuple[dict, bool]:
        """Wait for ``kernel``, download its output; (summary, all done?)."""
        status = self.wait(kernel, on_status=on_status)
        summary = self.download(job_id, kernel)
        if summary.get("complete") or (self._is_presenter_job(job_id) and status == "complete"):
            self.log("all shots are ready")
            return summary, True
        if status == "error" and not summary.get("done"):
            raise KaggleError(
                f"The Kaggle run failed before making any shot. Open https://www.kaggle.com/code/{kernel} "
                "and look at the log (Output / Logs).")
        self.log(f"{summary.get('done', 0)}/{summary.get('total', '?')} shots ready; starting another run")
        return summary, False

    # -- helpers --------------------------------------------------------------------
    def _is_presenter_job(self, job_id: str) -> bool:
        with open(os.path.join(job_package.job_dir(job_id), "package", "job.json"), encoding="utf-8") as fp:
            return json.load(fp).get("kind") == "create_presenter"

    def _wait_dataset(self, ref: str, timeout: float = 900) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                status = str(self.api.dataset_status(ref) or "").lower()
            except Exception as exc:
                status = f"error: {exc}"
            if "ready" in status:
                return
            self.sleep(10)
        raise KaggleError(f"Kaggle did not finish processing the dataset {ref}")

    @staticmethod
    def _check(response) -> None:
        error = getattr(response, "error", None)
        if error:
            raise KaggleError(str(error))

    def _state(self, job_id: str) -> dict:
        path = os.path.join(job_package.job_dir(job_id), "kaggle.json")
        try:
            with open(path, encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, ValueError):
            return {}

    def _save_state(self, job_id: str, **fields) -> None:
        state = self._state(job_id)
        state.update(fields)
        with open(os.path.join(job_package.job_dir(job_id), "kaggle.json"), "w", encoding="utf-8") as fp:
            json.dump(state, fp, indent=2)
