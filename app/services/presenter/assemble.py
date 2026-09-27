"""Assemble the final presenter video on the laptop.

1. every shot becomes a clip of exactly its narration length at the output
   resolution (Lanczos upscale from the worker's 480p) and 25 fps;
   missing shots fall back to a slow zoom on their first frame, so one failed
   shot never ruins the video;
2. clips are joined with short crossfades; each clip is extended by holding
   its last frame for the fade, so the timeline stays in sync with the
   narration;
3. the narration is one continuous track (the shots' audio back to back) and
   the per-shot subtitles are merged with the right offsets;
4. ``video.generate_video`` adds subtitles, music and the logo as usual, then
   the intro/outro is added.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from loguru import logger

from app.models.schema import VideoAspect, VideoParams
from app.services import branding, video
from app.services.presenter import package as job_package
from app.utils import utils

FPS = 25
FADE = 0.4


def _ffmpeg(args: list[str]) -> None:
    command = [utils.get_ffmpeg_binary(), "-y", "-loglevel", "error", *args]
    # FFmpeg writes UTF-8; never decode it with the Windows code page.
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-1500:]}")


def _scale_filter(width: int, height: int) -> str:
    return (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},setsar=1,unsharp=5:5:0.4")


def _still_clip(image: str, output: str, width: int, height: int, seconds: float) -> str:
    frames = max(1, round(seconds * FPS))
    fitted = output + ".png"
    from PIL import Image

    img = Image.open(image).convert("RGB")
    scale = max(width * 1.1 / img.width, height * 1.1 / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - round(width * 1.1)) // 2, (img.height - round(height * 1.1)) // 2
    img.crop((left, top, left + round(width * 1.1), top + round(height * 1.1))).save(fitted)
    try:
        _ffmpeg(["-loop", "1", "-i", fitted, "-vf",
                 f"zoompan=z='min(1+0.0007*on,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                 f":d={frames}:s={width}x{height}:fps={FPS},format=yuv420p",
                 "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", output])
    finally:
        os.remove(fitted)
    return output


def shot_source(job_root: str, package: str, shot: dict, presenter_ref: str, work: str,
                width: int, height: int) -> tuple[str, str]:
    """(clip path, kind): the rendered shot, a local screenshot clip or a still fallback."""
    rendered = os.path.join(job_root, "output", "shots", f"{shot['id']}.mp4")
    if os.path.isfile(rendered) and os.path.getsize(rendered) > 0:
        return rendered, "ai"
    duration = float(shot["duration"])
    target = os.path.join(work, f"{shot['id']}_local.mp4")
    location = os.path.join(package, shot["location_image"]) if shot.get("location_image") else ""
    if shot.get("local") and location:
        return branding.render_screenshot_clip(location, target, width, height, duration + FADE), "screen"
    frame = os.path.join(job_root, "output", "frames", f"{shot['id']}.png")
    still = next((p for p in (frame, location, presenter_ref) if p and os.path.isfile(p)), "")
    if not still:
        raise RuntimeError(f"shot {shot['id']} has no video and no image to fall back to")
    logger.warning(f"shot {shot['id']} was not rendered; using a still image instead")
    return _still_clip(still, target, width, height, duration + FADE), "still"


def join_clips(clips: list[str], durations: list[float], output: str, width: int, height: int) -> str:
    """Normalise and crossfade; the result lasts exactly sum(durations)."""
    inputs: list[str] = []
    filters: list[str] = []
    last = len(clips) - 1
    for i, (clip, seconds) in enumerate(zip(clips, durations)):
        inputs += ["-i", clip]
        length = seconds + (FADE if i < last else 0)
        filters.append(f"[{i}:v]{_scale_filter(width, height)},fps={FPS},format=yuv420p,"
                       f"tpad=stop_mode=clone:stop_duration={length:.3f},"
                       f"trim=duration={length:.3f},setpts=PTS-STARTPTS[v{i}]")
    current = "v0"
    offset = 0.0
    for i in range(1, len(clips)):
        offset += durations[i - 1]
        filters.append(f"[{current}][v{i}]xfade=transition=fade:duration={FADE}:offset={offset:.3f}[x{i}]")
        current = f"x{i}"
    script = output + ".filter.txt"
    with open(script, "w", encoding="utf-8") as fp:
        fp.write(";\n".join(filters))
    try:
        _ffmpeg([*inputs, "-filter_complex_script", script, "-map", f"[{current}]", "-an",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", output])
    finally:
        os.remove(script)
    return output


def join_audio(audio_files: list[str], durations: list[float], output: str) -> str:
    """One narration track: each shot's audio padded/trimmed to its shot length."""
    inputs: list[str] = []
    filters = []
    for i, (audio, seconds) in enumerate(zip(audio_files, durations)):
        if audio and os.path.isfile(audio):
            inputs += ["-i", audio]
        else:
            inputs += ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=44100:cl=mono"]
        filters.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=mono,apad,"
                       f"atrim=duration={seconds:.3f},asetpts=PTS-STARTPTS[a{i}]")
    filters.append("".join(f"[a{i}]" for i in range(len(durations))) + f"concat=n={len(durations)}:v=0:a=1[out]")
    _ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-c:a", "pcm_s16le", output])
    return output


_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _seconds(stamp: str) -> float:
    h, m, s, ms = _SRT_TIME.match(stamp.strip()).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def read_srt(path: str) -> list[tuple[float, float, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig") as fp:
        blocks = re.split(r"\n\s*\n", fp.read().replace("\r", "").strip())
    items = []
    for block in blocks:
        lines = block.split("\n")
        for index, line in enumerate(lines):
            if "-->" in line:
                start, end = line.split("-->")
                text = "\n".join(lines[index + 1:]).strip()
                if text:
                    items.append((_seconds(start), _seconds(end), text))
                break
    return items


def merge_subtitles(srt_files: list[str], durations: list[float], output: str) -> str:
    entries = []
    offset = 0.0
    for path, seconds in zip(srt_files, durations):
        for start, end, text in read_srt(path):
            entries.append((offset + min(start, seconds), offset + min(end, seconds), text))
        offset += seconds
    with open(output, "w", encoding="utf-8") as fp:
        for index, (start, end, text) in enumerate(entries, start=1):
            fp.write(f"{index}\n{utils.time_convert_seconds_to_hmsm(start)} --> "
                     f"{utils.time_convert_seconds_to_hmsm(end)}\n{text}\n\n")
    return output


def assemble(job_id: str, params: VideoParams, output_file: str | None = None, progress=None) -> str:
    root = job_package.job_dir(job_id)
    package = os.path.join(root, "package")
    with open(os.path.join(package, "job.json"), encoding="utf-8") as fp:
        job = json.load(fp)
    params = params.model_copy()
    params.video_aspect = VideoAspect(job["settings"].get("aspect", "16:9"))
    width, height = params.video_aspect.to_resolution()
    work = os.path.join(root, "assemble")
    os.makedirs(work, exist_ok=True)
    refs = job.get("presenter", {}).get("reference_images") or []
    presenter_ref = os.path.join(package, refs[0]) if refs else ""

    clips, durations, audio, subtitles = [], [], [], []
    report = {"ai": 0, "screen": 0, "still": 0}
    for shot in job["shots"]:
        clip, kind = shot_source(root, package, shot, presenter_ref, work, width, height)
        report[kind] += 1
        clips.append(clip)
        durations.append(float(shot["duration"]))
        audio.append(os.path.join(package, shot["audio"]) if shot.get("audio") else "")
        subtitles.append(os.path.join(root, "subtitles", f"{shot['id']}.srt"))
    if progress:
        progress(0.3)
    combined = join_clips(clips, durations, os.path.join(work, "combined.mp4"), width, height)
    if progress:
        progress(0.6)
    narration = join_audio(audio, durations, os.path.join(work, "narration.wav"))
    srt = merge_subtitles(subtitles, durations, os.path.join(work, "subtitles.srt")) if params.subtitle_enabled else ""
    final = output_file or os.path.join(root, "final.mp4")
    video.generate_video(combined, narration, srt, final, params)
    if params.add_intro_outro:
        branding.add_intro_outro(final)
    if progress:
        progress(1.0)
    logger.info(f"presenter video ready: {final} (shots: {report})")
    return final
