"""Optional intro / outro added around the finished video.

Put a file named ``intro.*`` and/or ``outro.*`` (video: mp4/mov/mkv/webm,
or a still image: png/jpg) in the branding folder, ``[app] branding_dir``
or ``<storage>/branding`` by default. Each part is re-encoded to exactly
match the final video (size, frame rate, audio format) and then joined
with a stream copy, so the long main video is never re-encoded.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from loguru import logger

from app.config import config
from app.utils import utils

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
IMAGE_SECONDS = 3.0
# MP4 time scale used by FFmpeg/libx264 for 30 fps output. Parts must share it
# with the main video for the concat demuxer to keep timestamps continuous.
_VIDEO_TIMESCALE = "15360"


def branding_dir(create: bool = False) -> str:
    configured = str(config.app.get("branding_dir", "") or "").strip()
    if configured:
        directory = os.path.abspath(os.path.expanduser(os.path.expandvars(configured)))
        if create:
            os.makedirs(directory, exist_ok=True)
        return directory
    return utils.storage_dir("branding", create=create)


def find_part(name: str, directory: str | None = None) -> str:
    """Return the first ``intro.*`` / ``outro.*`` file, or an empty string."""
    directory = directory or branding_dir()
    if not os.path.isdir(directory):
        return ""
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        stem, ext = os.path.splitext(entry)
        if (
            stem.lower() == name
            and ext.lower() in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS
            and os.path.isfile(path)
        ):
            return path
    return ""


def _probe(video_path: str) -> dict:
    from moviepy import VideoFileClip

    with VideoFileClip(video_path) as clip:
        info = {
            "width": int(clip.w),
            "height": int(clip.h),
            "fps": float(clip.fps or 30),
            "has_audio": clip.audio is not None,
            "sample_rate": 44100,
            "channels": 2,
        }
        if clip.audio is not None:
            info["sample_rate"] = int(clip.audio.fps or 44100)
            info["channels"] = int(getattr(clip.audio, "nchannels", 2) or 2)
    return info


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-800:] or "ffmpeg failed")


def _encode_part(source: str, output: str, target: dict) -> None:
    ffmpeg = utils.get_ffmpeg_binary()
    w, h, fps = target["width"], target["height"], target["fps"]
    is_image = source.lower().endswith(IMAGE_EXTENSIONS)
    has_audio = False if is_image else _probe(source)["has_audio"]

    command = [ffmpeg, "-y", "-loglevel", "error"]
    if is_image:
        command += ["-loop", "1", "-t", str(IMAGE_SECONDS), "-i", source]
    else:
        command += ["-i", source]
    if not has_audio:
        layout = "stereo" if target["channels"] >= 2 else "mono"
        command += ["-f", "lavfi", "-i", f"anullsrc=r={target['sample_rate']}:cl={layout}"]

    video_filter = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps:g},format=yuv420p"
    )
    if is_image:
        video_filter += f",fade=t=in:d=0.5,fade=t=out:st={IMAGE_SECONDS - 0.5}:d=0.5"
    command += [
        "-vf", video_filter,
        "-map", "0:v:0",
        "-map", "0:a:0" if has_audio else "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-ar", str(target["sample_rate"]), "-ac", str(target["channels"]),
        "-video_track_timescale", _VIDEO_TIMESCALE,
        "-shortest",
        output,
    ]
    _run(command)


def add_intro_outro(video_path: str, directory: str | None = None) -> str:
    """Wrap ``video_path`` in place with the intro/outro. Returns the path.

    Any failure is logged and the original video is kept untouched: branding
    is decoration and must never cost the user a finished long video.
    """
    intro = find_part("intro", directory)
    outro = find_part("outro", directory)
    if not intro and not outro:
        logger.info("no intro/outro files found in the branding folder, skipping")
        return video_path

    work_dir = os.path.dirname(video_path)
    temp_files = []
    try:
        target = _probe(video_path)
        if not target["has_audio"]:
            logger.warning("final video has no audio track, skip intro/outro")
            return video_path
        parts = []
        for name, source in (("intro", intro), ("outro", outro)):
            if not source:
                continue
            encoded = os.path.join(work_dir, f"branding-{name}.mp4")
            _encode_part(source, encoded, target)
            temp_files.append(encoded)
            parts.append((name, encoded))

        ordered = [p for n, p in parts if n == "intro"] + [video_path] + [
            p for n, p in parts if n == "outro"
        ]
        list_file = os.path.join(work_dir, "branding-concat.txt")
        temp_files.append(list_file)
        with open(list_file, "w", encoding="utf-8") as fp:
            for item in ordered:
                escaped = Path(item).resolve().as_posix().replace("'", "'\\''")
                fp.write(f"file '{escaped}'\n")
        output = os.path.join(work_dir, "branding-output.mp4")
        temp_files.append(output)
        _run(
            [
                utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", list_file,
                "-c", "copy", "-movflags", "+faststart", output,
            ]
        )
        os.replace(output, video_path)
        temp_files.remove(output)
        logger.success(
            f"added {' and '.join(name for name, _ in parts)} to {os.path.basename(video_path)}"
        )
    except Exception as exc:
        logger.error(f"failed to add intro/outro, keeping the video without them: {exc}")
    finally:
        for file in temp_files:
            try:
                os.remove(file)
            except OSError:
                pass
    return video_path
