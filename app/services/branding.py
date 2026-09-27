"""Branding: intro / outro, a logo watermark and website screenshots.

Everything lives in the branding folder, ``[app] branding_dir`` or
``<storage>/branding`` by default:

- ``intro.*`` / ``outro.*`` (video: mp4/mov/mkv/webm, or a still image:
  png/jpg). ``intro_portrait.*`` / ``outro_portrait.*`` are preferred for
  vertical videos. Each part is re-encoded to exactly match the final video
  (size, frame rate, audio format) and then joined with a stream copy, so the
  long main video is never re-encoded.
- ``logo.png`` (transparent background works best): shown in a corner of the
  whole video when "logo watermark" is enabled.
- ``screenshots/``: images of a website or product. Names containing
  "mobile" or "portrait" are used for vertical videos, the others for
  landscape ones. They become short slow-zoom clips mixed into the footage.
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


def find_part(
    name: str,
    directory: str | None = None,
    portrait: bool = False,
    extensions: tuple[str, ...] = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS,
) -> str:
    """Return ``<name>.*`` (``<name>_portrait.*`` first for vertical videos) or ""."""
    directory = directory or branding_dir()
    if not os.path.isdir(directory):
        return ""
    stems = [f"{name}_portrait", name] if portrait else [name]
    entries = sorted(os.listdir(directory))
    for wanted in stems:
        for entry in entries:
            path = os.path.join(directory, entry)
            stem, ext = os.path.splitext(entry)
            if stem.lower() == wanted and ext.lower() in extensions and os.path.isfile(path):
                return path
    return ""


# --------------------------------------------------------------------------- logo
def watermark_image(logo_path: str, target_width: int, output_path: str) -> str:
    """Logo on a soft rounded white plate, so dark logos stay readable on any footage."""
    from PIL import Image, ImageDraw

    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.crop(logo.getbbox() or (0, 0, *logo.size))
    scale = target_width / logo.width
    logo = logo.resize((max(1, round(logo.width * scale)), max(1, round(logo.height * scale))), Image.LANCZOS)
    pad_x, pad_y = max(6, target_width // 12), max(4, logo.height // 3)
    plate = Image.new("RGBA", (logo.width + 2 * pad_x, logo.height + 2 * pad_y), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle(
        (0, 0, plate.width - 1, plate.height - 1),
        radius=min(plate.height // 2, pad_x * 2),
        fill=(255, 255, 255, 215),
    )
    plate.alpha_composite(logo, (pad_x, pad_y))
    plate.save(output_path)
    return output_path


def watermark_clip(video_width: int, video_height: int, duration: float, work_dir: str):
    """MoviePy clip with the logo in the top corner, or None without logo.png."""
    logo = find_part("logo", extensions=IMAGE_EXTENSIONS)
    if not logo:
        logger.warning("logo watermark enabled but no logo.png in the branding folder")
        return None
    from moviepy import ImageClip

    portrait = video_height > video_width
    width = int(video_width * (0.30 if portrait else 0.16))
    margin = int(min(video_width, video_height) * 0.035)
    image = watermark_image(logo, width, os.path.join(work_dir, "watermark.png"))
    clip = ImageClip(image).with_duration(duration).with_opacity(0.92)
    return clip.with_position((video_width - clip.w - margin, margin))


# --------------------------------------------------------------------------- screenshots
def list_screenshots(portrait: bool, directory: str | None = None) -> list[str]:
    """Screenshots matching the video orientation (see module docstring)."""
    folder = os.path.join(directory or branding_dir(), "screenshots")
    if not os.path.isdir(folder):
        return []
    files = sorted(
        os.path.join(folder, entry)
        for entry in os.listdir(folder)
        if entry.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder, entry))
    )
    vertical = [f for f in files if any(k in os.path.basename(f).lower() for k in ("mobile", "portrait"))]
    horizontal = [f for f in files if f not in vertical]
    preferred = vertical if portrait else horizontal
    return preferred or files


def screenshot_frame(image_path: str, width: int, height: int, output_path: str) -> str:
    """Place a screenshot on a blurred copy of itself, with rounded corners and a shadow."""
    from PIL import Image, ImageDraw, ImageFilter

    shot = Image.open(image_path).convert("RGB")
    cover = max(width / shot.width, height / shot.height)
    background = shot.resize((int(shot.width * cover) + 1, int(shot.height * cover) + 1), Image.LANCZOS)
    left, top = (background.width - width) // 2, (background.height - height) // 2
    background = background.crop((left, top, left + width, top + height))
    background = background.filter(ImageFilter.GaussianBlur(radius=max(8, width // 60)))
    background = Image.blend(background, Image.new("RGB", (width, height), (15, 23, 42)), 0.55)

    max_w, max_h = int(width * 0.86), int(height * 0.86)
    scale = min(max_w / shot.width, max_h / shot.height)
    fitted = shot.resize((max(1, int(shot.width * scale)), max(1, int(shot.height * scale))), Image.LANCZOS)

    radius = max(8, width // 80)
    mask = Image.new("L", fitted.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, fitted.width - 1, fitted.height - 1), radius=radius, fill=255)
    x, y = (width - fitted.width) // 2, (height - fitted.height) // 2
    shadow = Image.new("L", (width, height), 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        (x, y + radius, x + fitted.width, y + fitted.height + radius), radius=radius, fill=150
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=radius * 2))
    background.paste(Image.new("RGB", (width, height), (0, 0, 0)), (0, 0), shadow)
    background.paste(fitted, (x, y), mask)
    background.save(output_path)
    return output_path


def render_screenshot_clip(image_path: str, output_file: str, width: int, height: int, seconds: float = 4.0) -> str:
    """Framed screenshot with a gentle zoom, sized exactly for the video canvas."""
    frame = screenshot_frame(image_path, width * 2, height * 2, output_file + ".png")
    fps = 30
    frames = int(seconds * fps)
    try:
        _run([
            utils.get_ffmpeg_binary(), "-y", "-loglevel", "error", "-loop", "1", "-i", frame,
            "-vf", (
                f"zoompan=z='min(1+0.0006*on,1.06)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={frames}:s={width}x{height}:fps={fps},format=yuv420p"
            ),
            "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            output_file,
        ])
    finally:
        try:
            os.remove(frame)
        except OSError:
            pass
    return output_file


def screenshot_clips(width: int, height: int, count: int, work_dir: str) -> list[str]:
    """Render up to ``count`` screenshot clips, spread over the available images."""
    images = list_screenshots(portrait=height > width)
    if not images or count <= 0:
        return []
    if len(images) > count:
        step = len(images) / count
        images = [images[int(i * step)] for i in range(count)]
    clips = []
    for index, image in enumerate(images, start=1):
        target = os.path.join(work_dir, f"screenshot-{index}.mp4")
        try:
            clips.append(render_screenshot_clip(image, target, width, height))
        except Exception as exc:
            logger.warning(f"skipping screenshot {os.path.basename(image)}: {exc}")
    return clips


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
    work_dir = os.path.dirname(video_path)
    temp_files = []
    try:
        target = _probe(video_path)
        portrait = target["height"] > target["width"]
        intro = find_part("intro", directory, portrait=portrait)
        outro = find_part("outro", directory, portrait=portrait)
        if not intro and not outro:
            logger.info("no intro/outro files found in the branding folder, skipping")
            return video_path
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
