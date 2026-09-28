"""Job package for the cloud worker (``kaggle/presenter_worker.py``).

``<storage>/presenter_jobs/<job_id>/``::

    package/            uploaded to Kaggle as a private dataset
        job.json
        presenter/ref1.png ...
        audio/s01.mp3 ...    narration of every shot (edge-tts)
        locations/*.png      location photos / website screenshots
        previous/            finished shots of an earlier run (resume)
    subtitles/s01.srt ...    per-shot subtitles, merged when assembling
    output/                  what the worker produced (downloaded)
    plan.json               the editable shot list
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time

from loguru import logger

from app.services import voice
from app.services.presenter.profiles import Presenter
from app.utils import utils

JOB_VERSION = 1


def jobs_root() -> str:
    root = utils.storage_dir("presenter_jobs")
    os.makedirs(root, exist_ok=True)
    return root


def new_job_id(topic: str = "") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (topic or "").lower()).strip("-")[:24]
    return f"{time.strftime('%Y%m%d-%H%M%S')}{'-' + slug if slug else ''}"


def job_dir(job_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id or ""):
        raise ValueError(f"bad job id: {job_id!r}")
    return os.path.join(jobs_root(), job_id)


def save_plan(job_id: str, plan: dict) -> str:
    folder = job_dir(job_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "plan.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(plan, fp, ensure_ascii=False, indent=2)
    return path


def load_plan(job_id: str) -> dict:
    with open(os.path.join(job_dir(job_id), "plan.json"), encoding="utf-8") as fp:
        return json.load(fp)


def list_jobs() -> list[str]:
    root = jobs_root()
    return sorted((d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "plan.json"))),
                  reverse=True)


def _copy_image(src: str, dst: str, max_side: int = 1600) -> None:
    from PIL import Image

    image = Image.open(src).convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image.save(dst)


def record_narration(shot: dict, audio_file: str, subtitle_file: str, voice_name: str, voice_rate: float,
                     tts=None) -> float:
    """TTS one shot; returns its duration in seconds."""
    tts = tts or voice.tts
    sub_maker = tts(text=shot["narration"], voice_name=voice_name, voice_rate=voice_rate, voice_file=audio_file)
    if sub_maker is None or not os.path.isfile(audio_file):
        raise RuntimeError(f"text to speech failed for shot {shot['id']}")
    duration = voice.get_audio_duration(audio_file) or voice.get_audio_duration(sub_maker)
    try:
        voice.create_subtitle(sub_maker=sub_maker, text=shot["narration"], subtitle_file=subtitle_file)
    except Exception as exc:  # subtitles are optional; the video still works
        logger.warning(f"subtitle for shot {shot['id']} failed: {exc}")
    return round(float(duration), 3)


def build_package(job_id: str, presenter: Presenter, shots: list[dict], settings: dict,
                  places: dict[str, dict] | None = None, tts=None, progress=None) -> str:
    """Record the narration and write ``package/`` for the worker. Returns the package folder.

    ``places`` maps a place name (as used in ``shot["place"]``) to
    ``{"path": image file, "screen": bool}``; screens are website screenshots
    shown on a wall screen next to the presenter.
    """
    places = places or {}
    root = job_dir(job_id)
    package = os.path.join(root, "package")
    for sub in ("presenter", "audio", "locations"):
        os.makedirs(os.path.join(package, sub), exist_ok=True)
    os.makedirs(os.path.join(root, "subtitles"), exist_ok=True)

    refs = []
    for index, src in enumerate(presenter.reference_images[:3], start=1):
        name = f"presenter/ref{index}.png"
        _copy_image(src, os.path.join(package, name))
        refs.append(name)

    location_files: dict[str, str] = {}
    job_shots = []
    for index, shot in enumerate(shots):
        audio_name = f"audio/{shot['id']}.mp3"
        audio_path = os.path.join(package, audio_name)
        srt_path = os.path.join(root, "subtitles", f"{shot['id']}.srt")
        if not (os.path.isfile(audio_path) and shot.get("duration")):
            spoken = record_narration(shot, audio_path, srt_path, presenter.voice_name, presenter.voice_rate, tts)
            # ``min_duration``: the storyboard length; the picture holds after the narration ends.
            shot["duration"] = round(max(spoken, float(shot.get("min_duration") or 0)), 3)
        entry = {k: shot[k] for k in ("id", "type", "narration", "location", "action", "camera", "duration")
                 if k in shot}
        entry["audio"] = audio_name
        for flag in ("green", "local"):  # green: keyed and composited locally; local: made on the laptop
            if shot.get(flag):
                entry[flag] = True
        if entry["type"] == "ANIMATE":
            # Body motion from the motion library (a tested driving clip, mirrored to the target side).
            from app.services.presenter import motion_library

            driving = f"motions/{shot['id']}.mp4"
            info = motion_library.prepare(shot["motion"], entry["duration"], os.path.join(package, driving))
            entry.update({"driving": driving, "motion": info["motion"], "pose_prompt": info["pose_prompt"],
                          "mirrored": info["mirrored"], "green": True})
        place = places.get(shot.get("place", ""))
        if place:
            if place["path"] not in location_files:
                name = f"locations/place{len(location_files) + 1:02d}.png"
                _copy_image(place["path"], os.path.join(package, name))
                location_files[place["path"]] = name
            entry["location_image"] = location_files[place["path"]]
            entry["screen"] = bool(place.get("screen"))
            if entry["screen"] and entry["type"] == "BROLL":
                # Website on its own: rendered on the laptop so the text stays sharp.
                entry["local"] = True
        job_shots.append(entry)
        if progress:
            progress(index + 1, len(shots))

    job = {
        "version": JOB_VERSION,
        "job_id": job_id,
        "kind": "video",
        "settings": {"aspect": settings.get("aspect", "16:9"), "seed": int(settings.get("seed", 2026)),
                     "language": settings.get("language", "")},
        "presenter": {"name": presenter.name, "description": presenter.description, "reference_images": refs},
        "shots": job_shots,
    }
    with open(os.path.join(package, "job.json"), "w", encoding="utf-8") as fp:
        json.dump(job, fp, ensure_ascii=False, indent=2)
    logger.info(f"presenter job {job_id}: {len(job_shots)} shots, "
                f"{sum(s['duration'] for s in job_shots):.1f}s narration")
    return package


def build_create_presenter_package(job_id: str, description: str, count: int = 4) -> str:
    package = os.path.join(job_dir(job_id), "package")
    os.makedirs(package, exist_ok=True)
    with open(os.path.join(package, "job.json"), "w", encoding="utf-8") as fp:
        json.dump({"version": JOB_VERSION, "job_id": job_id, "kind": "create_presenter",
                   "prompt": f"Photo of {description}", "count": int(count)}, fp, indent=2)
    return package


def stage_previous_output(job_id: str) -> int:
    """Put finished shots of the last run into ``package/previous`` so a rerun skips them."""
    root = job_dir(job_id)
    output = os.path.join(root, "output")
    previous = os.path.join(root, "package", "previous")
    count = 0
    if not os.path.isfile(os.path.join(output, "progress.json")):
        return 0
    for sub in ("shots", "frames"):
        os.makedirs(os.path.join(previous, sub), exist_ok=True)
        for name in os.listdir(os.path.join(output, sub)) if os.path.isdir(os.path.join(output, sub)) else []:
            shutil.copyfile(os.path.join(output, sub, name), os.path.join(previous, sub, name))
            count += sub == "shots"
    shutil.copyfile(os.path.join(output, "progress.json"), os.path.join(previous, "progress.json"))
    return count
