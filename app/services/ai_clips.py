"""AI-generated clips made on a free Kaggle GPU (see kaggle/README-ar.md).

The laptop has no usable GPU, so AI video is generated elsewhere:

1. ``build_scene_prompts`` turns the narration into one English visual prompt
   per paragraph (in script order) and the user pastes them into the Kaggle
   notebook, which renders ``001.mp4``, ``002.mp4``, ...;
2. the user unzips those clips into ``<storage>/ai_clips/<name>/``;
3. ``mix_ai_clips`` spreads them over the stock footage timeline, each clip
   close to the part of the narration its prompt was written for.
"""

from __future__ import annotations

import os
import re

from loguru import logger

from app.services import llm
from app.utils import file_security, utils

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".m4v")
MAX_PROMPTS = 60


def ai_clips_root(create: bool = True) -> str:
    return utils.storage_dir("ai_clips", create=create)


def resolve_ai_clips_folder(name: str) -> str:
    """Resolve a folder *name* inside ``<storage>/ai_clips``; reject anything else."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("AI clips folder name is empty")
    folder = file_security.resolve_path_within_directory(
        ai_clips_root(), name, require_file=False
    )
    if not os.path.isdir(folder):
        raise ValueError(f"AI clips folder does not exist: {folder}")
    return folder


def list_ai_clips(folder: str) -> list[str]:
    """Clips sorted by file name, which the notebook numbers in script order."""
    files = [
        os.path.join(folder, entry)
        for entry in os.listdir(folder)
        if entry.lower().endswith(VIDEO_EXTENSIONS) and not entry.startswith(".")
    ]

    def natural_key(path: str):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", os.path.basename(path))]

    return sorted((f for f in files if os.path.isfile(f)), key=natural_key)


def mix_ai_clips(stock_videos: list[str], ai_videos: list[str]) -> list[str]:
    """Spread AI clips evenly over the (script-ordered) stock clip list.

    AI clip ``k`` of ``m`` lands at the proportional position ``(k + 0.5) / m``
    of the timeline, so a clip made for paragraph 3 of 10 appears about 30%
    into the video. With no stock clips the AI clips are used on their own.
    """
    if not ai_videos:
        return list(stock_videos)
    if not stock_videos:
        return list(ai_videos)
    total = len(stock_videos)
    slots: dict[int, list[str]] = {}
    for k, clip in enumerate(ai_videos):
        position = min(total, int((k + 0.5) * total / len(ai_videos)))
        slots.setdefault(position, []).append(clip)
    mixed = []
    for index in range(total + 1):
        mixed.extend(slots.get(index, []))
        if index < total:
            mixed.append(stock_videos[index])
    return mixed


def split_paragraphs(script: str) -> list[str]:
    text = utils.remove_pause_tags(script or "")
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def build_prompt_request(subject: str, paragraph: str) -> str:
    return f"""You write prompts for an AI video generator (Wan 2.2).
Video subject: {subject}
Narration for this part of the video:
{paragraph[:1500]}

Write ONE prompt for a 5-second cinematic shot that illustrates this narration.
Rules:
- English, one sentence of 25-45 words.
- Describe the subject, the setting, the lighting and the camera movement (e.g. "slow dolly-in", "aerial drone shot").
- Realistic, filmable scenes. No text, captions, logos, watermarks, split screens or famous people.
- Output only the prompt, nothing else."""


def clean_prompt(text: str) -> str:
    text = llm._THINK_BLOCK_RE.sub("", text or "")
    text = re.sub(r"[`*#]+", "", text)
    text = re.sub(r"^\s*(prompt|shot)\s*\d*\s*[:：-]\s*", "", text.strip(), flags=re.IGNORECASE)
    text = " ".join(text.split()).strip(" \"'")
    return text


def build_scene_prompts(subject: str, script: str, per_paragraph: int = 1, app_config=None) -> list[str]:
    """One (or more) English visual prompts per narration paragraph, in order."""
    per_paragraph = max(1, min(int(per_paragraph or 1), 3))
    prompts: list[str] = []
    for paragraph in split_paragraphs(script):
        for _ in range(per_paragraph):
            if len(prompts) >= MAX_PROMPTS:
                break
            request = build_prompt_request(subject, paragraph)
            if prompts:
                request += f'\n- Make it clearly different from this previous shot: "{prompts[-1]}"'
            response = (
                llm._generate_response(request)
                if app_config is None
                else llm._generate_response(request, app_config=app_config)
            )
            if not response or response.startswith("Error: "):
                raise RuntimeError(str(response or "empty response").removeprefix("Error: "))
            prompt = clean_prompt(response)
            if prompt:
                prompts.append(prompt)
    logger.info(f"built {len(prompts)} AI scene prompts")
    return prompts


def prompts_to_text(prompts: list[str]) -> str:
    """One prompt per line: the format the Kaggle notebook reads."""
    return "\n".join(p.replace("\n", " ").strip() for p in prompts if p.strip()) + "\n"
