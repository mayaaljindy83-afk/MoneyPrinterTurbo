"""Presenter motion library: tested driving clips for Wan Animate 2, reused instead of re-rolled.

Wan Animate 2 copies body motion, hands and face from a *real* driving video onto
the presenter's reference photo (no skeleton step). A gesture is therefore a short
clip of a person doing exactly that gesture; once a clip gives a good result it is
marked ``tested`` and reused for every video.

Sides are SCREEN sides (what the viewer sees): POINT_LEFT points towards the left
edge of the frame, where the target card must be. A missing side is made by
mirroring the other one (a horizontally flipped clip points the other way), so one
recording covers both.

Storage: ``<data>/motions/<MOTION>/drive.mp4`` + ``meta.json`` (on E:, never C:).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from app.utils import utils

FPS = 30  # Wan Animate 2 works on 30 fps driving video
SIZE = (480, 832)  # portrait: the presenter fills the height (sharper than 832x480 landscape)

# motion -> defaults. target_side / safe position are screen sides; "" = no target.
CATALOG: dict[str, dict] = {
    "IDLE_PRESENT": {"target_side": "", "duration": 3.0, "safe_presenter_position": "right", "camera_safe": True,
                     "pose_prompt": "a woman standing and talking calmly with small natural hand movements"},
    "TALK_CAMERA": {"target_side": "", "duration": 4.0, "safe_presenter_position": "right", "camera_safe": True,
                    "pose_prompt": "a woman talking to the camera with natural hand gestures"},
    "POINT_LEFT": {"target_side": "left", "duration": 3.0, "safe_presenter_position": "right", "camera_safe": True,
                   "pose_prompt": "a woman turns and points with her arm to the left side of the frame"},
    "POINT_RIGHT": {"target_side": "right", "duration": 3.0, "safe_presenter_position": "left", "camera_safe": True,
                    "pose_prompt": "a woman turns and points with her arm to the right side of the frame"},
    "POINT_UP": {"target_side": "up", "duration": 3.0, "safe_presenter_position": "right", "camera_safe": True,
                 "pose_prompt": "a woman points upwards above her"},
    "POINT_DOWN": {"target_side": "down", "duration": 3.0, "safe_presenter_position": "right", "camera_safe": True,
                   "pose_prompt": "a woman points down in front of her"},
    "PRESENT_LEFT_CARD": {"target_side": "left", "duration": 3.5, "safe_presenter_position": "right",
                          "camera_safe": True,
                          "pose_prompt": "a woman presents something on the left side of the frame with an open palm"},
    "PRESENT_RIGHT_CARD": {"target_side": "right", "duration": 3.5, "safe_presenter_position": "left",
                           "camera_safe": True,
                           "pose_prompt": "a woman presents something on the right side of the frame with an open palm"},
    "LOOK_LEFT": {"target_side": "left", "duration": 2.0, "safe_presenter_position": "right", "camera_safe": True,
                  "pose_prompt": "a woman turns her head and looks to the left side of the frame"},
    "LOOK_RIGHT": {"target_side": "right", "duration": 2.0, "safe_presenter_position": "left", "camera_safe": True,
                   "pose_prompt": "a woman turns her head and looks to the right side of the frame"},
    "TURN_LEFT": {"target_side": "left", "duration": 2.5, "safe_presenter_position": "right", "camera_safe": True,
                  "pose_prompt": "a woman turns her body towards the left side of the frame"},
    "TURN_RIGHT": {"target_side": "right", "duration": 2.5, "safe_presenter_position": "left", "camera_safe": True,
                   "pose_prompt": "a woman turns her body towards the right side of the frame"},
    "WALK_LEFT": {"target_side": "left", "duration": 4.0, "safe_presenter_position": "right", "camera_safe": False,
                  "pose_prompt": "a woman walks towards the left side of the frame"},
    "WALK_RIGHT": {"target_side": "right", "duration": 4.0, "safe_presenter_position": "left", "camera_safe": False,
                   "pose_prompt": "a woman walks towards the right side of the frame"},
    "WALK_FORWARD": {"target_side": "", "duration": 4.0, "safe_presenter_position": "center", "camera_safe": False,
                     "pose_prompt": "a woman walks towards the camera"},
    "WALK_AND_STOP": {"target_side": "", "duration": 4.0, "safe_presenter_position": "right", "camera_safe": False,
                      "pose_prompt": "a woman walks a few steps, stops and faces the camera"},
    "WELCOME": {"target_side": "", "duration": 3.0, "safe_presenter_position": "right", "camera_safe": True,
                "pose_prompt": "a woman opens her arms in a welcoming gesture and smiles"},
    "CTA_GESTURE": {"target_side": "", "duration": 3.0, "safe_presenter_position": "right", "camera_safe": True,
                    "pose_prompt": "a woman makes an inviting gesture towards the camera"},
}

MIRROR = {"left": "right", "right": "left"}


class MotionError(RuntimeError):
    pass


def mirror_name(motion: str) -> str:
    """POINT_LEFT <-> POINT_RIGHT, PRESENT_LEFT_CARD <-> PRESENT_RIGHT_CARD, ... ('' if none)."""
    for side, other in (("LEFT", "RIGHT"), ("RIGHT", "LEFT")):
        if side in motion.split("_"):
            name = "_".join(other if part == side else part for part in motion.split("_"))
            return name if name in CATALOG else ""
    return ""


def metadata(motion: str, **overrides) -> dict:
    """The motion's metadata, e.g. {"motion": "POINT_LEFT", "target_side": "left", "duration": 3.0,
    "safe_presenter_position": "right", "camera_safe": true, ...}."""
    if motion not in CATALOG:
        raise MotionError(f"unknown motion {motion!r}; known: {', '.join(CATALOG)}")
    meta = {"motion": motion, **CATALOG[motion]}
    meta.update({k: v for k, v in overrides.items() if v is not None})
    return meta


def root() -> str:
    folder = utils.storage_dir("motions")
    os.makedirs(folder, exist_ok=True)
    return folder


def clip_path(motion: str) -> str:
    return os.path.join(root(), motion, "drive.mp4")


def load_meta(motion: str) -> dict:
    try:
        with open(os.path.join(root(), motion, "meta.json"), encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return {}


def _save_meta(motion: str, data: dict) -> None:
    folder = os.path.join(root(), motion)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def _ffmpeg(args: list[str]) -> None:
    result = subprocess.run([utils.get_ffmpeg_binary(), "-loglevel", "error", "-y", *args],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise MotionError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")


def import_clip(motion: str, source: str, source_note: str = "", tested: bool = False) -> str:
    """Add a driving clip (a phone recording, a stock clip, a generated clip) for ``motion``.

    Normalised once: 30 fps, portrait 480x832 (centre crop), no audio, H.264.
    """
    metadata(motion)
    if not os.path.isfile(source):
        raise MotionError(f"clip not found: {source}")
    target = clip_path(motion)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    w, h = SIZE
    _ffmpeg(["-i", source, "-an", "-vf", f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase,"
             f"crop={w}:{h},setsar=1", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", target + ".tmp.mp4"])
    os.replace(target + ".tmp.mp4", target)
    _save_meta(motion, {**metadata(motion), "source": source_note or os.path.basename(source), "tested": tested})
    return target


def mark_tested(motion: str, tested: bool = True, notes: str = "") -> None:
    data = load_meta(motion) or metadata(motion)
    data["tested"] = tested
    if notes:
        data["notes"] = notes
    _save_meta(motion, data)


def available() -> dict[str, dict]:
    """Every motion with how it can be made now: own clip, mirrored from its twin, or missing."""
    status = {}
    for motion in CATALOG:
        twin = mirror_name(motion)
        if os.path.isfile(clip_path(motion)):
            status[motion] = {"source": "clip", "tested": bool(load_meta(motion).get("tested"))}
        elif twin and os.path.isfile(clip_path(twin)):
            status[motion] = {"source": f"mirror of {twin}", "tested": bool(load_meta(twin).get("tested"))}
        else:
            status[motion] = {"source": "", "tested": False}
    return status


def prepare(motion: str, duration: float, output: str) -> dict:
    """Driving clip for one shot: the motion's clip (or its mirrored twin), cut or held to ``duration``.

    Returns the metadata written into the job (motion, pose prompt, mirrored, frames).
    """
    meta = metadata(motion)
    source, mirrored = clip_path(motion), False
    if not os.path.isfile(source):
        twin = mirror_name(motion)
        if twin and os.path.isfile(clip_path(twin)):
            source, mirrored = clip_path(twin), True
        else:
            raise MotionError(f"No driving clip for {motion} yet. Add one in the Motion Library "
                              f"(a {meta['duration']:.0f}-second clip of a person doing it).")
    frames = max(1, round(float(duration) * FPS))
    filters = ["hflip"] if mirrored else []
    # Hold the last frame when the clip is shorter than the shot (the gesture ends and stays).
    filters.append(f"tpad=stop_mode=clone:stop_duration={float(duration):.3f}")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    _ffmpeg(["-i", source, "-an", "-vf", ",".join(filters), "-frames:v", str(frames), "-r", str(FPS),
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", output])
    source_meta = load_meta(os.path.basename(os.path.dirname(source)))
    return {"motion": motion, "pose_prompt": meta["pose_prompt"], "mirrored": mirrored, "frames": frames,
            "fps": FPS, "tested": bool(source_meta.get("tested")), "target_side": meta["target_side"]}


def copy_library(destination: str) -> None:
    """Backup/export of all clips (e.g. to move the library to another drive)."""
    shutil.copytree(root(), destination, dirs_exist_ok=True)


# --------------------------------------------------------------------------- AI candidates (option B)
DRIVE_STYLE = ("the whole upper body and both hands stay inside the frame, clear deliberate gesture, "
               "plain light grey studio background, static camera, knees-up framing")

# Free stock clips (option C): Pexels videos are free to use; search pages per motion.
PEXELS_SEARCH = "https://www.pexels.com/search/videos/{}/?orientation=portrait"
SEARCH_TERMS = {
    "IDLE_PRESENT": "woman presenting talking", "TALK_CAMERA": "woman talking to camera",
    "POINT_LEFT": "woman pointing side", "POINT_RIGHT": "woman pointing side", "POINT_UP": "woman pointing up",
    "POINT_DOWN": "woman pointing down", "PRESENT_LEFT_CARD": "woman presenting open palm",
    "PRESENT_RIGHT_CARD": "woman presenting open palm", "LOOK_LEFT": "woman looking side",
    "LOOK_RIGHT": "woman looking side", "TURN_LEFT": "woman turning", "TURN_RIGHT": "woman turning",
    "WALK_LEFT": "woman walking side", "WALK_RIGHT": "woman walking side", "WALK_FORWARD": "woman walking towards camera",
    "WALK_AND_STOP": "woman walking stops", "WELCOME": "woman welcoming gesture", "CTA_GESTURE": "woman inviting gesture",
}


def search_link(motion: str) -> str:
    from urllib.parse import quote

    return PEXELS_SEARCH.format(quote(SEARCH_TERMS.get(motion, motion.lower().replace("_", " "))))


def build_candidates_job(job_id: str, presenter, motions: list[str], variants: int = 2, seed: int = 2026) -> dict:
    """A cloud job making ``variants`` AI driving clips per motion from the presenter photo (Wan I2V).

    Returns the plan (shot id -> motion). Mirror twins are skipped: one side is enough.
    """
    from app.services.presenter import package as job_package

    wanted = []
    for motion in motions:
        metadata(motion)
        twin = mirror_name(motion)
        if twin and twin in wanted:
            continue
        wanted.append(motion)
    root = job_package.job_dir(job_id)
    package = os.path.join(root, "package")
    os.makedirs(os.path.join(package, "presenter"), exist_ok=True)
    refs = []
    for index, src in enumerate(presenter.reference_images[:1], start=1):
        name = f"presenter/ref{index}.png"
        job_package._copy_image(src, os.path.join(package, name))
        refs.append(name)
    shots, mapping = [], {}
    for motion in wanted:
        meta = metadata(motion)
        for variant in range(max(1, int(variants))):
            shot_id = f"m{len(shots) + 1:02d}"
            mapping[shot_id] = motion
            shots.append({"id": shot_id, "type": "DRIVE", "duration": float(meta["duration"]),
                          "action": f"{meta['pose_prompt']}, {DRIVE_STYLE}", "seed": seed + 1000 * variant + len(shots)})
    job = {"version": job_package.JOB_VERSION, "job_id": job_id, "kind": "video",
           "settings": {"aspect": "9:16", "seed": seed},
           "presenter": {"name": presenter.name, "description": presenter.description, "reference_images": refs},
           "shots": shots}
    with open(os.path.join(package, "job.json"), "w", encoding="utf-8") as fp:
        json.dump(job, fp, ensure_ascii=False, indent=2)
    plan = {"job_id": job_id, "kind": "motion_candidates", "presenter": presenter.name, "motions": mapping}
    job_package.save_plan(job_id, plan)
    return plan


def candidates(job_id: str) -> list[dict]:
    """Finished AI candidates of a job: [{"motion", "path", "shot"}]."""
    from app.services.presenter import package as job_package

    plan = job_package.load_plan(job_id)
    folder = os.path.join(job_package.job_dir(job_id), "output", "shots")
    found = []
    for shot_id, motion in sorted(plan.get("motions", {}).items()):
        path = os.path.join(folder, f"{shot_id}.mp4")
        if os.path.isfile(path):
            found.append({"motion": motion, "path": path, "shot": shot_id})
    return found


def use_candidate(motion: str, path: str) -> str:
    """Keep an AI candidate as the motion's driving clip (not yet 'tested' until a render proves it)."""
    return import_clip(motion, path, source_note="AI candidate (Wan I2V)", tested=False)


def delete(motion: str) -> None:
    shutil.rmtree(os.path.join(root(), motion), ignore_errors=True)
