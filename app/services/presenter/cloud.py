"""Which GPU service renders presenter shots: free Kaggle (default) or paid RunPod Serverless."""

from __future__ import annotations

import json
import os

from app.config import config
from app.services.presenter import kaggle_agent, runpod_agent
from app.services.presenter import package as job_package

PROVIDERS = ("kaggle", "runpod")


def provider() -> str:
    name = str(config.app.get("presenter_cloud", "kaggle") or "kaggle").strip().lower()
    return name if name in PROVIDERS else "kaggle"


def label() -> str:
    return "RunPod" if provider() == "runpod" else "Kaggle"


def ready() -> bool:
    """The chosen service has its credentials."""
    if provider() == "runpod":
        return bool(runpod_agent.configured_key() and runpod_agent.configured_endpoint())
    return bool(kaggle_agent.configured_token())


def last_run_provider(job_id: str) -> str:
    """The service of this job's latest cloud run ("" if none)."""
    root = job_package.job_dir(job_id)
    runs = []
    for name, key, service in (("kaggle.json", "kernel", "kaggle"), ("runpod.json", "request", "runpod")):
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as fp:
                if json.load(fp).get(key):
                    runs.append((os.path.getmtime(path), service))
        except (OSError, ValueError):
            continue
    return max(runs)[1] if runs else ""


def make_agent(token: str = "", log=None, job_id: str = ""):
    """The agent of the chosen service. With ``job_id`` (continuing a job), the service that
    ran it last, so Continue fetches from where the work is instead of starting a new run."""
    service = (last_run_provider(job_id) if job_id else "") or provider()
    if service == "runpod":
        return runpod_agent.RunPodAgent(log=log)
    return kaggle_agent.KaggleAgent(token=token or kaggle_agent.configured_token(), log=log)


def has_run(job_id: str) -> bool:
    """A cloud run (on either service) was started for this job, so its results can be fetched."""
    return bool(last_run_provider(job_id))
