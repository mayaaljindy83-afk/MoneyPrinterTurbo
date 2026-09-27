"""Long-form (1-10 minute) narration scripts.

Small local models (4B parameters on a laptop CPU) lose the thread and start
repeating themselves when asked for 1,500 words in one go. Instead we:

1. ask for a short outline (one title per section);
2. write every section in its own request, giving the model the outline and
   the end of the previous section so the narration flows;
3. ask for a few stock-footage search terms per section, in order, so the
   footage follows the narration and many different clips are used.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from loguru import logger

from app.services import llm
from app.utils import rtl_text

MIN_MINUTES = 0.5
MAX_MINUTES = 10.0
# Target length of one section. Around 45-70 seconds of narration keeps each
# request short enough for a 4B model while giving the footage time to breathe.
WORDS_PER_SECTION = 130
MAX_SECTIONS = 14
TERMS_PER_SECTION = 3
_MAX_ATTEMPTS = 3

# Measured narration speed of edge-tts neural voices at rate 1.0. Arabic words
# are longer (clitics are attached), so fewer words fit in a minute.
_WORDS_PER_MINUTE = {"ar": 115, "fa": 120, "ur": 125, "he": 120}
_DEFAULT_WORDS_PER_MINUTE = 150

_LANGUAGE_NAMES = {
    "ar": "Modern Standard Arabic (فصحى)",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "tr": "Turkish",
}


@dataclass
class LongScript:
    script: str
    sections: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


def _language_code(language: str | None, subject: str) -> str:
    code = str(language or "").strip().lower().replace("_", "-").split("-")[0]
    if code:
        return code
    return "ar" if rtl_text.contains_rtl(subject) else ""


def describe_language(language: str | None, subject: str) -> str:
    code = _language_code(language, subject)
    if not code:
        return "the same language as the video subject"
    return _LANGUAGE_NAMES.get(code, language or code)


def words_per_minute(language: str | None, subject: str = "", voice_rate: float = 1.0) -> int:
    code = _language_code(language, subject)
    base = _WORDS_PER_MINUTE.get(code, _DEFAULT_WORDS_PER_MINUTE)
    try:
        rate = float(voice_rate or 1.0)
    except (TypeError, ValueError):
        rate = 1.0
    rate = min(max(rate, 0.5), 2.0)
    return max(1, int(round(base * rate)))


def clamp_minutes(minutes) -> float:
    try:
        value = float(minutes)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return min(max(value, MIN_MINUTES), MAX_MINUTES)


def plan_sections(minutes: float, language: str | None, subject: str = "", voice_rate: float = 1.0):
    """Return (section_count, words_per_section) for the requested duration."""
    total_words = clamp_minutes(minutes) * words_per_minute(language, subject, voice_rate)
    count = max(1, min(MAX_SECTIONS, round(total_words / WORDS_PER_SECTION)))
    return count, max(40, int(round(total_words / count)))


def _is_error(response) -> bool:
    return not response or (isinstance(response, str) and response.startswith("Error: "))


def _parse_json_list(response: str) -> list[str]:
    text = llm._strip_code_fence(response or "")
    candidates = [text]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidates.append(match.group())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    return []


_HEADING_RE = re.compile(
    r"^\s*(section|part|chapter|القسم|الجزء|المقطع|الفقرة)\b.*?[:：]\s*$", re.IGNORECASE
)


def clean_section(text: str) -> str:
    """Strip markdown, headings and stage directions a model may add anyway."""
    text = llm._THINK_BLOCK_RE.sub("", text or "")
    lines = []
    for line in text.replace("\r", "").split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _HEADING_RE.match(stripped) or stripped.startswith("#"):
            continue
        stripped = re.sub(r"[*_#`]+", "", stripped)
        stripped = re.sub(r"\[.*?\]", "", stripped)
        stripped = re.sub(r"^\s*(narrator|voiceover|الراوي|المعلق)\s*[:：]\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"^\s*[-•]\s+", "", stripped)
        if stripped.strip():
            lines.append(stripped.strip())
    # Paragraphs inside a section are joined; sections are separated by a
    # blank line by the caller.
    return " ".join(line for line in lines if line).strip()


def build_outline_prompt(subject: str, language_name: str, count: int, extra: str = "") -> str:
    prompt = f"""You are planning the narration of a YouTube video.
Video subject: {subject}
Number of sections: {count}

Return ONLY a JSON array of exactly {count} short section titles (3-8 words each), written in {language_name}.
The first section is a hook that grabs attention, the middle sections each cover one distinct idea, and the last section is a conclusion.
Do not repeat ideas between sections. Do not add numbering or any text outside the JSON array."""
    if extra:
        prompt += f"\n\nAdditional requirements from the user:\n{extra}"
    return prompt


def build_section_prompt(
    subject: str,
    language_name: str,
    titles: list[str],
    index: int,
    words: int,
    previous_tail: str = "",
    extra: str = "",
) -> str:
    count = len(titles)
    outline = "\n".join(f"{i + 1}. {title}" for i, title in enumerate(titles))
    if index == 0:
        position = (
            "This is the opening section: start with a strong hook sentence. "
            "Do not say 'welcome' or introduce yourself."
        )
    elif index == count - 1:
        position = (
            "This is the final section: wrap up the main ideas and end with a short, "
            "memorable closing sentence."
        )
    else:
        position = "This is a middle section: do not introduce the video again and do not conclude."
    if count == 1:
        position = "This is the whole script: open with a hook and end with a short conclusion."

    prompt = f"""Write section {index + 1} of {count} of a voice-over script for a video.
Video subject: {subject}
Full outline:
{outline}

Write ONLY section {index + 1}: "{titles[index]}".
{position}

Rules:
- Write about {words} words in {language_name}.
- Spoken narration only, in natural flowing sentences. No title, no heading, no bullet points, no markdown, no emojis.
- Do not write stage directions, speaker names, or words like "narrator".
- Do not repeat facts that belong to other sections.
- Output only the narration text."""
    if previous_tail:
        prompt += f"\n\nThe previous section ended with:\n\"{previous_tail}\"\nContinue naturally from there."
    if extra:
        prompt += f"\n\nAdditional requirements from the user:\n{extra}"
    return prompt


def build_terms_prompt(subject: str, section_text: str, amount: int) -> str:
    return f"""Suggest {amount} different stock-video search terms for this part of a narrated video.
Video subject: {subject}
Narration:
{section_text[:1500]}

Rules:
- English only, 1-3 words each, concrete things a camera can film (e.g. "city traffic night", "doctor laptop").
- Each term must show something different, to avoid repetitive footage.
- Return ONLY a JSON array of strings."""


def _ask(prompt: str, app_config=None) -> str:
    last = ""
    for attempt in range(_MAX_ATTEMPTS):
        response = (
            llm._generate_response(prompt)
            if app_config is None
            else llm._generate_response(prompt, app_config=app_config)
        )
        if not _is_error(response) and response.strip():
            return response
        last = response or "empty response"
        logger.warning(f"llm request failed (attempt {attempt + 1}/{_MAX_ATTEMPTS}): {last[:200]}")
    raise RuntimeError(str(last).removeprefix("Error: ").strip() or "llm request failed")


def _last_sentences(text: str, max_chars: int = 300) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    cut = re.search(r"[.!?。؟!]\s", tail)
    return tail[cut.end():].strip() if cut else tail.strip()


def generate_outline(subject: str, language_name: str, count: int, extra: str = "", app_config=None) -> list[str]:
    if count == 1:
        return [subject]
    try:
        titles = _parse_json_list(_ask(build_outline_prompt(subject, language_name, count, extra), app_config))
    except RuntimeError as exc:
        logger.warning(f"outline generation failed, using a generic outline: {exc}")
        titles = []
    if len(titles) < count:
        # A weak model may return fewer titles; keep the plan length anyway.
        titles += [f"{subject} ({i + 1})" for i in range(len(titles), count)]
    return titles[:count]


def generate_section_terms(subject: str, section_text: str, amount: int = TERMS_PER_SECTION, app_config=None) -> list[str]:
    try:
        terms = _parse_json_list(_ask(build_terms_prompt(subject, section_text, amount), app_config))
    except RuntimeError as exc:
        logger.warning(f"search terms generation failed for a section: {exc}")
        terms = []
    cleaned = []
    for term in terms:
        term = re.sub(r"\s+", " ", term).strip(" .\"'")
        # Search terms must be English for Pexels/Pixabay.
        if term and not rtl_text.contains_rtl(term) and term.lower() not in {t.lower() for t in cleaned}:
            cleaned.append(term)
    return cleaned[:amount]


def generate_long_script(
    subject: str,
    language: str | None,
    minutes: float,
    voice_rate: float = 1.0,
    extra_prompt: str = "",
    with_terms: bool = True,
    app_config=None,
    progress=None,
) -> LongScript:
    """Write a narration script of roughly ``minutes`` minutes, section by section."""
    language_name = describe_language(language, subject)
    count, words = plan_sections(minutes, language, subject, voice_rate)
    logger.info(
        f"long script: subject={subject!r}, minutes={clamp_minutes(minutes)}, "
        f"sections={count}, words_per_section={words}, language={language_name}"
    )
    titles = generate_outline(subject, language_name, count, extra_prompt, app_config)
    logger.info(f"outline: {titles}")

    sections: list[str] = []
    terms: list[str] = []
    for index in range(count):
        previous_tail = _last_sentences(sections[-1]) if sections else ""
        prompt = build_section_prompt(
            subject, language_name, titles, index, words, previous_tail, extra_prompt
        )
        text = clean_section(_ask(prompt, app_config))
        if not text:
            raise RuntimeError(f"section {index + 1} is empty")
        sections.append(text)
        logger.info(f"section {index + 1}/{count} written: {len(text.split())} words")
        if with_terms:
            section_terms = generate_section_terms(subject, text, app_config=app_config)
            for term in section_terms:
                if term.lower() not in {t.lower() for t in terms}:
                    terms.append(term)
        if progress:
            progress(index + 1, count)

    return LongScript(
        script="\n\n".join(sections), sections=sections, titles=titles, terms=terms
    )
