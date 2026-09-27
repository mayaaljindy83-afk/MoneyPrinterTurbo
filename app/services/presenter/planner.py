"""Split a narration script into presenter shots.

The narration is never rewritten here: it is cut at sentence boundaries into
shots of about 6-14 seconds. The local LLM only decides how every shot looks
(shot type, place, action, camera), so a small model cannot lose or invent
narration. If the model fails, a fixed rhythm of shot types is used.

Shot types:
- TALK   the presenter talks to the camera (lip-synced);
- POINT  the presenter talks and points at a screen / something in the place;
- WALK   the presenter walks through the place (voice-over, no lip-sync);
- BROLL  the place alone, without the presenter (voice-over).
"""

from __future__ import annotations

import json
import re

from loguru import logger

from app.services import llm
from app.services.long_script import words_per_minute

SHOT_TYPES = ("TALK", "POINT", "WALK", "BROLL")
MIN_SECONDS = 6.0
TARGET_SECONDS = 10.0
MAX_SECONDS = 14.0
BATCH = 12
# Used when the model gives no usable answer: talking shots dominate, with a
# walk or a cut-away every few shots so the video does not feel static.
_RHYTHM = ("TALK", "WALK", "TALK", "POINT", "BROLL", "TALK", "WALK", "POINT")
_DEFAULT_LOOK = {
    "TALK": ("talks to the camera with natural hand gestures", "medium shot, eye level"),
    "POINT": ("points at the screen beside her while explaining", "medium wide shot"),
    "WALK": ("walks slowly towards the camera, looking around", "tracking shot"),
    "BROLL": ("", "slow cinematic pan"),
}

_SENTENCE_RE = re.compile(r"[^.!?؟!\n]+[.!?؟!]*", re.UNICODE)


def split_sentences(script: str) -> list[str]:
    sentences = []
    for part in _SENTENCE_RE.findall(script or ""):
        part = part.strip()
        if part.strip(".!?؟! "):
            sentences.append(part)
    return sentences


def estimate_seconds(text: str, language: str, voice_rate: float = 1.0) -> float:
    words = len(text.split())
    return words * 60.0 / words_per_minute(language, text, voice_rate)


def chunk_narration(script: str, language: str, voice_rate: float = 1.0) -> list[str]:
    """Group sentences into chunks of roughly TARGET_SECONDS of speech."""
    chunks: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(script):
        seconds = estimate_seconds(" ".join(current + [sentence]), language, voice_rate)
        if current and seconds > MAX_SECONDS:
            chunks.append(" ".join(current))
            current = []
        current.append(sentence)
        if estimate_seconds(" ".join(current), language, voice_rate) >= TARGET_SECONDS:
            chunks.append(" ".join(current))
            current = []
    if current:
        tail = " ".join(current)
        if chunks and estimate_seconds(tail, language, voice_rate) < MIN_SECONDS / 2:
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)
    return chunks


def default_shot(index: int, total: int, narration: str, place: str = "") -> dict:
    if index == 0 or index == total - 1:
        kind = "TALK"
    else:
        kind = _RHYTHM[index % len(_RHYTHM)]
    action, camera = _DEFAULT_LOOK[kind]
    return {"id": f"s{index + 1:02d}", "type": kind, "narration": narration,
            "location": place or "a bright modern studio", "action": action, "camera": camera}


def build_look_prompt(topic: str, chunks: list[str], places: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(chunks))
    place_hint = ("Real places or screens you can use (write the name exactly): " + "; ".join(places) + "\n"
                  if places else "")
    return (
        "You are directing a video with one female presenter who explains a topic.\n"
        f"Topic: {topic}\n{place_hint}"
        "For every numbered narration part below, choose how the shot looks.\n"
        "Shot types: TALK (she talks to the camera), POINT (she talks and points at a screen or object), "
        "WALK (she walks through the place), BROLL (the place only, without her).\n"
        "Mostly TALK, with WALK/POINT/BROLL for variety. The first and last parts must be TALK.\n"
        "Answer ONLY with a JSON array, one object per part, in order, in English:\n"
        '[{"type": "TALK", "location": "short visual description of the place", '
        '"action": "what she does", "camera": "shot size and camera move"}]\n\n'
        f"Narration parts:\n{numbered}\n"
    )


def parse_looks(text: str, count: int) -> list[dict]:
    match = re.search(r"\[.*\]", llm._THINK_BLOCK_RE.sub("", text or ""), re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    looks = []
    for item in items[:count]:
        if not isinstance(item, dict):
            return []
        kind = str(item.get("type", "")).strip().upper()
        looks.append({
            "type": kind if kind in SHOT_TYPES else "",
            "location": str(item.get("location", "") or "")[:200],
            "action": str(item.get("action", "") or "")[:200],
            "camera": str(item.get("camera", "") or "")[:100],
        })
    return looks


def _match_place(location: str, places: list[str]) -> str:
    lowered = location.lower()
    for place in places:
        if place.lower() in lowered or lowered in place.lower():
            return place
    return ""


def plan_shots(script: str, topic: str, language: str, voice_rate: float = 1.0,
               places: list[str] | None = None, generate=None) -> list[dict]:
    """Return shot dicts: id, type, narration, location, action, camera[, place].

    ``places`` are names of location photos / screenshots the user supplied;
    a shot whose location mentions one gets ``place`` set to that name.
    """
    places = places or []
    generate = generate or llm._generate_response
    chunks = chunk_narration(script, language, voice_rate)
    shots = [default_shot(i, len(chunks), text, places[i % len(places)] if places else "")
             for i, text in enumerate(chunks)]
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        try:
            looks = parse_looks(generate(build_look_prompt(topic, batch, places)), len(batch))
        except Exception as exc:  # the default rhythm still gives a usable plan
            logger.warning(f"shot planner: LLM failed, using default shots: {exc}")
            looks = []
        for offset, look in enumerate(looks):
            shot = shots[start + offset]
            if look["type"]:
                shot["type"] = look["type"]
            for key in ("location", "action", "camera"):
                if look[key]:
                    shot[key] = look[key]
    for index, shot in enumerate(shots):
        if index in (0, len(shots) - 1) and shot["type"] != "TALK":
            shot["type"] = "TALK"
        if shot["type"] == "BROLL":
            shot["action"] = ""
        place = _match_place(shot["location"], places)
        if place:
            shot["place"] = place
        elif shot["type"] == "BROLL":
            # Without a real photo of the place, show her walking through it instead.
            shot["type"] = "WALK"
            shot["action"], shot["camera"] = _DEFAULT_LOOK["WALK"]
    logger.info(f"shot planner: {len(shots)} shots " + " ".join(s["type"] for s in shots))
    return shots
