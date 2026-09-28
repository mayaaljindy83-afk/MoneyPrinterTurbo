"""Narration with separate DISPLAY and SPOKEN text.

The voice engine reads the spoken form (brand names spelled out, optional tashkeel);
subtitles keep the display text exactly, timed with the voice's own word timings.
"""

from __future__ import annotations

import os

from app.config import config
from app.services.speech.pronunciation import PronunciationProcessor, llm_diacritizer


def language_of(voice_name: str) -> str:
    return "ar" if str(voice_name or "").lower().startswith("ar") else "en"


def processor_for(voice_name: str, engine: str = "edge", overrides: dict | None = None) -> PronunciationProcessor:
    """Edge and SILMA both read digits themselves (SILMA through its NeMo text normaliser), so numbers
    stay digits for them; other engines get them spelled out. LLM tashkeel is opt-in for Edge
    (``pronunciation_tashkeel`` / the Voice Lab); SILMA always adds its own (CATT)."""
    language = language_of(voice_name)
    tashkeel = bool(config.app.get("pronunciation_tashkeel", False)) and language == "ar" and engine == "edge"
    numbers = "keep" if engine in ("edge", "silma") else "words"
    return PronunciationProcessor(language, numbers=numbers, overrides=overrides,
                                  diacritizer=llm_diacritizer if tashkeel else None)


def split_lines(text: str) -> list[str]:
    """Subtitle lines, split the same way the rest of the program splits them."""
    from app.services import voice
    from app.utils import utils

    return [line for line in utils.split_string_by_punctuations(voice._format_text(text)) if line.strip()]


def prepare(display: str, processor: PronunciationProcessor) -> dict:
    """{"lines": [(display line, spoken line)], "spoken": full spoken text, "replacements", "unknown_terms"}."""
    pairs, replacements, unknown = [], [], set()
    for line in split_lines(display) or [display]:
        result = processor.process(line)
        pairs.append((line, result.spoken))
        replacements += result.replacements
        unknown.update(result.unknown_terms)
    return {"lines": pairs, "spoken": " ".join(spoken for _, spoken in pairs), "replacements": replacements,
            "unknown_terms": sorted(unknown)}


def word_times(sub_maker) -> list[tuple[float, float]]:
    """(start, end) seconds of every word the engine reported."""
    cues = getattr(sub_maker, "cues", None)
    if cues:
        return [(c.start.total_seconds(), c.end.total_seconds()) for c in cues if str(c.content).strip()]
    offsets = getattr(sub_maker, "offset", None) or []
    return [(a / 1e7, b / 1e7) for a, b in offsets]


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_display_srt(sub_maker, lines: list[tuple[str, str]], subtitle_file: str) -> bool:
    """Subtitles showing the DISPLAY lines, timed by the words of the matching SPOKEN lines.

    Words are shared out by each spoken line's word count, scaled to the number of timed
    words the engine returned (engines count punctuation / numbers slightly differently).
    """
    times = word_times(sub_maker)
    counts = [max(1, len(spoken.split())) for _, spoken in lines]
    if not times or not lines:
        return False
    total, entries, done = sum(counts), [], 0
    for index, ((display, _), count) in enumerate(zip(lines, counts)):
        first = min(len(times) - 1, int(done / total * len(times)))
        done += count
        last = max(first, min(len(times) - 1, int(round(done / total * len(times))) - 1))
        start, end = times[first][0], times[last][1]
        entries.append(f"{index + 1}\n{_srt_time(start)} --> {_srt_time(max(end, start + 0.2))}\n{display.strip()}\n")
    os.makedirs(os.path.dirname(subtitle_file) or ".", exist_ok=True)
    with open(subtitle_file, "w", encoding="utf-8") as fp:
        fp.write("\n".join(entries))
    return True
