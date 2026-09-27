"""Website ad pipeline: analyse -> storyboard -> presenter job (Kaggle) -> composited video.

A project lives in ``<storage>/marketing/<project_id>/``::

    project.json   inputs, the website folder, the ad plan (storyboard), preview/full job ids
    website/       website.json + real screenshots (website.py)

Rendering reuses the presenter studio: a preview or full video is a normal
presenter job (``<storage>/presenter_jobs/<job_id>``) whose plan has
``kind = "marketing"``, so packaging, the Kaggle agent, "Continue" after the
laptop was off and finished-shot skipping all work as before. Only the final
assembly differs: scenes are composited here.

Which scene needs the cloud GPU:
    TALK           presenter talking in an AI-made place (full frame)       -> Kaggle
    WEBSITE_WORLD  presenter talking on green, composited with the page      -> Kaggle + laptop
    POINT, CTA     same as WEBSITE_WORLD with other layouts                  -> Kaggle + laptop
    AI_SCENE       text-to-image + Wan motion, no presenter                  -> Kaggle
    WEBSITE        the real page, camera over it                              -> laptop only
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import urllib.parse

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.services import branding, video
from app.services.long_script import words_per_minute
from app.services.marketing import compositor, director, website
from app.services.presenter import assemble, profiles, studio
from app.services.presenter import package as job_package
from app.utils import utils

PREVIEW_SECONDS = 10
PRESENTER_SCENES = {"WEBSITE_WORLD": "world", "POINT": "point", "CTA": "cta"}
_resolution_lock = threading.Lock()


# --------------------------------------------------------------------------- projects
def projects_root() -> str:
    return utils.storage_dir("marketing", create=True)


def project_dir(project_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", project_id or ""):
        raise ValueError(f"bad project id: {project_id!r}")
    return os.path.join(projects_root(), project_id)


def save_project(project: dict) -> dict:
    folder = project_dir(project["project_id"])
    os.makedirs(folder, exist_ok=True)
    project["updated"] = time.time()
    tmp = os.path.join(folder, "project.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(project, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(folder, "project.json"))
    return project


def load_project(project_id: str) -> dict:
    with open(os.path.join(project_dir(project_id), "project.json"), encoding="utf-8") as fp:
        return json.load(fp)


def list_projects() -> list[str]:
    root = projects_root()
    return sorted((d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "project.json"))),
                  reverse=True)


def analyze(url: str, language: str, duration: float, platform: str, goal: str, presenter: str = "",
            aspect: str = "", focus: str = "", generate=None, read=None) -> dict:
    """Read the page, write the ad plan, save the project. ``read``/``generate`` are for tests."""
    url = website.normalize_url(url)
    host = re.sub(r"[^a-z0-9]+", "-", urllib.parse.urlparse(url).netloc.lower()).strip("-")
    project_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{host}"[:60]
    folder = project_dir(project_id)
    site_dir = os.path.join(folder, "website")
    site = (read or website.read_website)(url, site_dir)
    plan = director.direct(site, language, duration, platform, goal, focus, generate=generate)
    project = {
        "project_id": project_id, "created": time.time(), "url": url,
        "language": director.language_code(language), "duration": float(duration), "platform": platform,
        "goal": goal, "presenter": presenter or profiles.default_presenter(),
        "aspect": aspect or director.platform_aspect(platform), "focus": focus,
        "website_dir": site_dir, "plan": plan, "jobs": {},
    }
    save_project(project)
    logger.info(f"marketing project {project_id}: {len(plan['scenes'])} scenes, fallback={plan['fallback']}")
    return project


def update_scenes(project_id: str, scenes: list[dict]) -> dict:
    """Save edits from the storyboard table (validated again against the page)."""
    project = load_project(project_id)
    site = website.load_website(project["website_dir"])
    plan = dict(project["plan"], scenes=scenes)
    checked = director.validate_plan(plan, site, project["language"], project["duration"])
    project["plan"].update(checked)
    project["plan"]["voiceover"] = " ".join(s["voiceover"] for s in checked["scenes"])
    project["jobs"].pop("full", None)  # the storyboard changed: the full video must be made again
    return save_project(project)


# --------------------------------------------------------------------------- scenes -> shots
def _trim_words(text: str, seconds: float, code: str) -> str:
    words = text.split()
    limit = max(3, int(seconds * words_per_minute(code) / 60))
    if len(words) <= limit:
        return text
    first = re.split(r"(?<=[.!?؟])\s+", text)[0]
    if len(first.split()) <= limit:
        return first
    return " ".join(words[:limit]).rstrip("،,;:") + "."


def preview_scenes(plan: dict, site: dict) -> list[dict]:
    """~10 s: the presenter talking inside the website world, then the real page."""
    code = plan.get("language", "ar")
    scenes = plan["scenes"]
    hero = next((s for s in scenes if s["type"] in ("WEBSITE_WORLD", "POINT")), None) or \
        next((s for s in scenes if s["type"] in ("TALK", "CTA")), scenes[0])
    first = dict(hero, id="s01", type="WEBSITE_WORLD", duration=6.5,
                 voiceover=_trim_words(hero["voiceover"], 6.5, code))
    if not first.get("website_asset") or first["website_asset"] not in {s["id"] for s in site["screenshots"]}:
        first["website_asset"] = "hero"
    closing = next((s for s in reversed(scenes) if s["type"] == "CTA"), scenes[-1])
    second = {"id": "s02", "type": "WEBSITE", "voiceover": _trim_words(closing["voiceover"], 3.0, code),
              "website_asset": "page" if any(s["id"] == "page" for s in site["screenshots"]) else "hero",
              "presenter_action": "", "visual_prompt": "", "duration": 3.5, "claims": closing.get("claims", [])}
    return [first, second]


def scene_to_shot(scene: dict) -> dict:
    kind = scene["type"]
    shot = {"id": scene["id"], "narration": scene["voiceover"],
            "location": scene.get("visual_prompt") or "a modern bright studio lit in the brand colours",
            "action": scene.get("presenter_action") or "talks to the camera with natural hand gestures",
            "camera": "medium shot, eye level", "scene_type": kind, "website_asset": scene.get("website_asset", ""),
            "min_duration": float(scene.get("duration") or 0)}
    if kind == "TALK":
        shot["type"] = "TALK"
    elif kind in PRESENTER_SCENES:
        shot["type"] = "TALK"
        shot["green"] = True
        shot["camera"] = "medium shot, knees up"
    elif kind == "AI_SCENE":
        shot["type"] = "AI_SCENE"
    else:  # WEBSITE
        shot["type"] = "BROLL"
        shot["local"] = True
    return shot


def create_job(project_id: str, mode: str) -> str:
    """Make the presenter job (preview or full) for a project; returns its job id."""
    project = load_project(project_id)
    if not project.get("presenter"):
        raise ValueError("Choose a presenter first (Presenter Video page).")
    site = website.load_website(project["website_dir"])
    scenes = preview_scenes(project["plan"], site) if mode == "preview" else project["plan"]["scenes"]
    job_id = job_package.new_job_id(f"{mode}-{project_id[-20:]}")
    shots = [scene_to_shot(scene) for scene in scenes]
    job_package.save_plan(job_id, {
        "job_id": job_id, "kind": "marketing", "mode": mode, "project_id": project_id,
        "topic": project["plan"].get("service_name", ""), "language": project["language"],
        "aspect": project["aspect"], "presenter": project["presenter"], "places": [], "shots": shots,
        "scenes": scenes, "website_dir": project["website_dir"], "created": time.time()})
    project["jobs"][mode] = job_id
    save_project(project)
    studio.set_status(job_id, "planned", f"{mode} job with {len(shots)} scenes")
    return job_id


def start(project_id: str, mode: str, token: str, options: dict | None = None, agent=None) -> str:
    """Create (or reuse the unfinished) job and render it on Kaggle in the background."""
    project = load_project(project_id)
    job_id = project["jobs"].get(mode)
    if job_id and (studio.can_continue(job_id) or studio.read_status(job_id).get("state") in studio.ACTIVE_STATES):
        studio.start_resume(job_id, token, options, agent=agent)  # fetch; never redo finished scenes
        return job_id
    job_id = create_job(project_id, mode)
    studio.start_kaggle_render(job_id, token, options, agent=agent)
    return job_id


# --------------------------------------------------------------------------- local render
@contextlib.contextmanager
def _resolution(value: str):
    with _resolution_lock:
        old = config.app.get("video_resolution", "")
        config.app["video_resolution"] = value
        try:
            yield
        finally:
            config.app["video_resolution"] = old


def _shot_path(site: dict, asset: str) -> str:
    for shot in site["screenshots"]:
        if shot["id"] == asset:
            return os.path.join(site["_dir"], shot["path"])
    return ""


def _cards_for(site: dict, scene: dict, count: int) -> list[str]:
    cards = [s for s in site["screenshots"] if s["kind"] == "card"]
    chosen = [s for s in cards if s["id"] == scene.get("website_asset")]
    fact_text = {t["id"]: t["text"] for t in site["texts"]}
    claimed = " ".join(fact_text.get(c, "") for c in scene.get("claims", []))
    chosen += [s for s in cards if s not in chosen and s.get("text") and s["text"] in claimed]
    chosen += [s for s in cards if s not in chosen]
    offset = int(re.sub(r"\D", "", scene.get("id", "0")) or 0)  # vary the cards between scenes
    if len(chosen) > count and not scene.get("website_asset", "").startswith("card"):
        chosen = chosen[offset % len(chosen):] + chosen[: offset % len(chosen)]
    return [os.path.join(site["_dir"], s["path"]) for s in chosen[:count]]


def scene_spec(scene: dict, site: dict, portrait: bool, rtl: bool) -> dict:
    """What the compositor draws for a scene: real screenshots only."""
    kind = scene["type"]
    asset = scene.get("website_asset", "")
    screen = _shot_path(site, asset) if asset in ("hero", "page", "mobile") else ""
    if portrait and asset in ("hero", "page", "") and _shot_path(site, "mobile"):
        screen = _shot_path(site, "mobile") if kind != "WEBSITE" else _shot_path(site, "page")
    if not screen:
        screen = _shot_path(site, "page" if kind == "WEBSITE" else "hero")
    spec = {"style": PRESENTER_SCENES.get(kind, "screen" if kind == "WEBSITE" else "world"),
            "colors": site["brand"].get("colors", []), "screen": screen, "rtl": rtl,
            "seed": int(re.sub(r"\D", "", scene.get("id", "0")) or 0)}
    if kind == "POINT":
        spec["cards"] = _cards_for(site, scene, 1)
    elif kind == "WEBSITE_WORLD":
        spec["cards"] = _cards_for(site, scene, 2)
    elif kind == "CTA":
        spec["cta"] = _shot_path(site, asset) if asset.startswith("cta") else _shot_path(site, "cta1")
        spec["logo"] = _shot_path(site, "logo") or branding.find_part("logo")
    return spec


def render(job_id: str, options: dict | None = None) -> str:
    """Composite every scene, join them with the narration, add subtitles, music and branding."""
    root = job_package.job_dir(job_id)
    plan = job_package.load_plan(job_id)
    with open(os.path.join(root, "package", "job.json"), encoding="utf-8") as fp:
        job = json.load(fp)
    site = website.load_website(plan["website_dir"])
    scenes = {s["id"]: s for s in plan["scenes"]}
    preview = plan.get("mode") == "preview"
    aspect = VideoAspect(plan["aspect"])
    rtl = str(plan.get("language", "")).startswith("ar")
    refs = job.get("presenter", {}).get("reference_images") or []
    presenter_photo = os.path.join(root, "package", refs[0]) if refs else ""
    work = os.path.join(root, "assemble")
    os.makedirs(work, exist_ok=True)
    studio.set_status(job_id, "assembling", "building the website world scenes")

    with _resolution("720p" if preview else config.app.get("video_resolution", "") or "1080p"):
        width, height = aspect.to_resolution()
        clips, durations, audio, subtitles = [], [], [], []
        for index, shot in enumerate(job["shots"]):
            scene = scenes.get(shot["id"], {"id": shot["id"], "type": shot.get("scene_type", "WEBSITE")})
            seconds = float(shot["duration"])
            cloud = os.path.join(root, "output", "shots", f"{shot['id']}.mp4")
            has_cloud = os.path.isfile(cloud) and os.path.getsize(cloud) > 0
            kind = scene["type"]
            if kind in ("TALK", "AI_SCENE") and has_cloud:
                clip = cloud
            else:
                spec = scene_spec(scene, site, height > width, rtl)
                if kind in PRESENTER_SCENES or kind == "TALK":
                    spec["presenter_video" if has_cloud else "presenter_image"] = cloud if has_cloud else presenter_photo
                    if not has_cloud:
                        logger.warning(f"scene {shot['id']}: no rendered presenter, using her photo")
                clip = compositor.render_scene(spec, os.path.join(work, f"{shot['id']}_scene.mp4"), width, height,
                                               seconds + assemble.FADE, preset="veryfast" if preview else "medium")
            clips.append(clip)
            durations.append(seconds)
            audio.append(os.path.join(root, "package", shot["audio"]))
            subtitles.append(os.path.join(root, "subtitles", f"{shot['id']}.srt"))
            studio.set_status(job_id, progress=0.9 + 0.06 * (index + 1) / len(job["shots"]))
        combined = assemble.join_clips(clips, durations, os.path.join(work, "combined.mp4"), width, height)
        narration = assemble.join_audio(audio, durations, os.path.join(work, "narration.wav"))
        params = studio.default_params(plan, options)
        params.video_aspect = aspect
        srt = assemble.merge_subtitles(subtitles, durations, os.path.join(work, "subtitles.srt")) \
            if params.subtitle_enabled else ""
        final = os.path.join(root, "preview.mp4" if preview else "final.mp4")
        video.generate_video(combined, narration, srt, final, params)
    if params.add_intro_outro and not preview:
        branding.add_intro_outro(final)
    studio.set_status(job_id, "done", "the video is ready", final=final, progress=1.0)
    logger.info(f"marketing video ready: {final}")
    return final
