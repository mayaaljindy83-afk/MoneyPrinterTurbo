"""PronunciationProcessor: turns what is SHOWN into what is SPOKEN, before any TTS.

The display text (subtitles, website text) is never changed; only the copy sent
to the voice engine is. Handles, for Arabic and English:

* a pronunciation dictionary (brand names, acronyms, terms) with per-project overrides;
* numbers, decimals, percentages and years -> words (for engines that misread digits);
* punctuation that should become a pause;
* optional Arabic tashkeel from a diacritizer, accepted only if it changed nothing
  but the marks (the words themselves can never be altered).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

# Harakat, tanween, shadda, sukun, dagger alif.
TASHKEEL = re.compile("[\u064b-\u0652\u0670]")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٪", "0123456789.%")
ARABIC_LETTER = re.compile("[\u0621-\u064a]")

# Official brand pronunciation: letter by letter, in every engine and language. Protected:
# user and project dictionaries cannot change it, and the display text always stays "QAI-VO".
QAI_VO = {"ar": "كيو إيه آي ڤي أو", "en": "Q A I V O"}
# How the TEXT sent to a voice can be written so that it SOUNDS like the official pronunciation
# (e.g. Edge's Arabic voice cannot say "ڤ"). Chosen by ear in the Voice Lab, saved in config.toml
# as qai_vo_spoken_ar / qai_vo_spoken_en; the display text is always "QAI-VO".
BRAND_VARIANTS = {
    "ar": ["كيو إيه آي ڤي أو", "كيو إيه آي في أو", "كيو إيه آي فِي أُو", "كيو، إيه، آي، في، أو",
           "Q A I V O"],
    "en": ["Q A I V O", "Q.A.I. V.O.", "Q-A-I V-O", "Queue A I Vee Oh"],
}


def brand_spoken(language: str) -> str:
    try:
        from app.config import config

        chosen = str(config.app.get(f"qai_vo_spoken_{language}", "") or "").strip()
    except Exception:
        chosen = ""
    return chosen or QAI_VO[language]


PROTECTED = {
    "qai-vo.com": {"ar": QAI_VO["ar"] + " دوت كوم", "en": QAI_VO["en"] + " dot com"},
    "QAI-VO": QAI_VO,
    "QAI VO": QAI_VO,
    "QAIVO": QAI_VO,
    "QAI_VO": QAI_VO,
}

# Default spoken forms. Keys match case-insensitively on word boundaries; longer keys win.
DEFAULT_DICTIONARY = {
    # English voices read these as written (the Voice Lab showed that spelling them out sounds worse).
    "DOI": {"ar": "دي أو آي"},
    "AI": {"ar": "إيه آي"},
    "API": {"ar": "إيه بي آي"},
    "APA": {"ar": "إيه بي إيه"},
    "MLA": {"ar": "إم إل إيه"},
    "PDF": {"ar": "بي دي إف"},
    "OpenAI": {"ar": "أوبن إيه آي"},
    "Gemini": {"ar": "جيميناي"},
    "Academic Suite": {"ar": "أكاديميك سويت"},
}

_ONES = ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
_TEENS = {11: "أحد عشر", 12: "اثنا عشر"}
_TENS = {2: "عشرون", 3: "ثلاثون", 4: "أربعون", 5: "خمسون", 6: "ستون", 7: "سبعون", 8: "ثمانون", 9: "تسعون"}
_HUNDREDS = {1: "مئة", 2: "مئتان", 3: "ثلاثمئة", 4: "أربعمئة", 5: "خمسمئة", 6: "ستمئة", 7: "سبعمئة",
             8: "ثمانمئة", 9: "تسعمئة"}
_SCALES_AR = [(1_000_000_000, ("مليار", "ملياران", "مليارات", "مليارًا")),
              (1_000_000, ("مليون", "مليونان", "ملايين", "مليونًا")),
              (1_000, ("ألف", "ألفان", "آلاف", "ألفًا"))]

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


PUNCTUATION = ".,!?؟،؛:…()[]\"'«»“”"


def _core(word: str) -> str:
    return word.strip(PUNCTUATION)


def strip_tashkeel(text: str) -> str:
    return TASHKEEL.sub("", text)


def _ar_below_1000(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100
    if n:
        if n < 10:
            parts.append(_ONES[n])
        elif n == 10:
            parts.append("عشرة")
        elif n < 20:
            parts.append(_TEENS.get(n) or f"{_ONES[n - 10]} عشر")
        else:
            ones, tens = n % 10, n // 10
            parts.append(f"{_ONES[ones]} و{_TENS[tens]}" if ones else _TENS[tens])
    return " و".join(parts)


def arabic_number(n: int) -> str:
    """Standard Arabic (masculine, nominative) words for a whole number."""
    if n < 0:
        return "سالب " + arabic_number(-n)
    if n == 0:
        return _ONES[0]
    parts = []
    for scale, (one, two, plural, accusative) in _SCALES_AR:
        count, n = divmod(n, scale)
        if not count:
            continue
        if count == 1:
            parts.append(one)
        elif count == 2:
            parts.append(two)
        elif count <= 10:
            parts.append(f"{_ar_below_1000(count)} {plural}")
        else:
            parts.append(f"{_ar_below_1000(count)} {accusative}")
    if n:
        parts.append(_ar_below_1000(n))
    return " و".join(parts)


def _en_below_1000(n: int) -> str:
    words = []
    if n >= 100:
        words.append(f"{_EN_ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        words.append(_EN_TENS[n // 10] + (f"-{_EN_ONES[n % 10]}" if n % 10 else ""))
    elif n or not words:
        words.append(_EN_ONES[n])
    return " ".join(words)


def english_number(n: int) -> str:
    if n < 0:
        return "minus " + english_number(-n)
    if n < 1000:
        return _en_below_1000(n)
    words = []
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")):
        count, n = divmod(n, scale)
        if count:
            words.append(f"{_en_below_1000(count)} {name}")
    if n:
        words.append(_en_below_1000(n))
    return " ".join(words)


def english_year(n: int) -> str:
    if 2000 <= n < 2010 or n % 100 == 0 or not 1100 <= n < 2100:
        return english_number(n)
    return f"{_en_below_1000(n // 100)} {_en_below_1000(n % 100)}" if n % 100 >= 10 else \
        f"{_en_below_1000(n // 100)} oh {_EN_ONES[n % 100]}"


@dataclass
class SpokenText:
    display: str
    spoken: str
    language: str
    replacements: list[dict] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)  # Latin words in Arabic text with no entry
    diacritized: bool = False

    def to_dict(self) -> dict:
        return {"display": self.display, "spoken": self.spoken, "language": self.language,
                "replacements": self.replacements, "unknown_terms": self.unknown_terms,
                "diacritized": self.diacritized}


class PronunciationProcessor:
    """``process(display_text)`` -> SpokenText. Options:

    * ``numbers``: "words" (spell out; best for F5/SILMA-style models) or "keep" (the engine
      reads digits itself, as Edge TTS does well);
    * ``overrides``: per-project entries ``{"term": "spoken"}`` or ``{"term": {"ar": .., "en": ..}}``;
    * ``diacritizer``: callable(text) -> text with tashkeel (Arabic only; optional).
    """

    def __init__(self, language: str = "ar", numbers: str = "words", overrides: dict | None = None,
                 dictionary: dict | None = None, diacritizer=None):
        self.language = "ar" if str(language).lower().startswith("ar") else "en"
        self.numbers = numbers
        self.diacritizer = diacritizer
        entries = dict(DEFAULT_DICTIONARY if dictionary is None else dictionary)
        entries.update(load_user_dictionary())
        entries.update(overrides or {})
        protected = {term.lower() for term in PROTECTED}
        entries = {term: spoken for term, spoken in entries.items() if term.lower() not in protected}
        entries.update(PROTECTED)
        brand = brand_spoken(self.language)
        self.entries = {}
        for term, spoken in entries.items():
            if spoken is QAI_VO:
                spoken = brand
            elif term == "qai-vo.com":
                spoken = brand + (" دوت كوم" if self.language == "ar" else " dot com")
            elif isinstance(spoken, dict):
                spoken = spoken.get(self.language) or ""
            if term and spoken:
                self.entries[term] = str(spoken)

    # -- steps ------------------------------------------------------------------
    def _apply_dictionary(self, text: str, replacements: list) -> str:
        for term in sorted(self.entries, key=len, reverse=True):
            pattern = re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.IGNORECASE)
            spoken = self.entries[term]

            def swap(match, term=term, spoken=spoken):
                replacements.append({"display": match.group(0), "spoken": spoken, "term": term})
                return f"\u2063{spoken}\u2063"  # invisible marks: never re-matched, removed later

            text = pattern.sub(swap, text)
        return text

    def _number_words(self, text: str) -> str:
        if self.numbers != "words":
            return text
        arabic = self.language == "ar"
        say = arabic_number if arabic else english_number

        def percent(m):
            return f"{self._decimal(m.group(1), say)} {'بالمئة' if arabic else 'percent'}"

        def number(m):
            raw = m.group(0)
            if "." in raw:
                return self._decimal(raw, say)
            value = int(raw.replace(",", ""))
            if not arabic and 1100 <= value < 2100 and "," not in raw and len(raw) == 4:
                return english_year(value)
            return say(value)

        text = re.sub(r"(\d+(?:\.\d+)?)\s*%", percent, text)
        return re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", number, text)

    @staticmethod
    def _decimal(raw: str, say) -> str:
        whole, _, fraction = raw.replace(",", "").partition(".")
        words = say(int(whole))
        if fraction:
            point = "فاصلة" if say is arabic_number else "point"
            words += f" {point} " + (say(int(fraction)) if say is arabic_number
                                     else " ".join(_EN_ONES[int(d)] for d in fraction))
        return words

    @staticmethod
    def _pauses(text: str) -> str:
        text = re.sub(r"\s*[—–]\s*", "، ", text)  # dashes read as a short pause
        text = re.sub(r"\.{3,}|…", "… ", text)
        text = re.sub(r"\s*[|•·]\s*", "، ", text)
        text = re.sub(r"[\"“”«»]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?؟…،,":
            text += "."
        return text

    def _diacritize(self, text: str) -> tuple[str, bool]:
        if self.language != "ar" or not self.diacritizer or not ARABIC_LETTER.search(text):
            return text, False
        try:
            marked = str(self.diacritizer(text) or "")
        except Exception:
            return text, False
        # Accept only if nothing but tashkeel changed: same words in the same order. Punctuation
        # the model dropped or moved is ignored; the original punctuation (pauses) is kept.
        words, marked_words = text.split(), marked.split()
        marked_words = [w for w in marked_words if _core(w)]  # a lone "." the model split off
        if len(marked_words) != len([w for w in words if _core(w)]):
            return text, False
        out, it = [], iter(marked_words)
        for word in words:
            core = _core(word)
            if not core:
                out.append(word)
                continue
            marked_core = _core(next(it))
            if strip_tashkeel(marked_core) != strip_tashkeel(core):
                return text, False
            out.append(word.replace(core, marked_core, 1))
        return " ".join(out), True

    def process(self, display: str) -> SpokenText:
        text = (display or "").translate(ARABIC_DIGITS)
        replacements: list[dict] = []
        text = self._apply_dictionary(text, replacements)
        text = self._number_words(text)
        text = self._pauses(text)
        unknown = []
        if self.language == "ar":
            # Latin words left outside dictionary entries: flagged so the user can add them.
            outside = re.sub("\u2063[^\u2063]*\u2063", " ", text)
            unknown = sorted(set(re.findall(r"[A-Za-z][A-Za-z0-9+-]*(?:\.[A-Za-z0-9]+)*", outside)))
        text, diacritized = self._diacritize(text.replace("\u2063", ""))
        return SpokenText(display=display, spoken=re.sub(r"\s+", " ", text).strip(), language=self.language,
                          replacements=replacements, unknown_terms=unknown, diacritized=diacritized)


def user_dictionary_path() -> str:
    from app.utils import utils

    folder = utils.storage_dir("voice")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "pronunciation.json")


def load_user_dictionary() -> dict:
    """The user's own entries (E:\\MoneyPrinterData\\voice\\pronunciation.json), if any."""
    try:
        with open(user_dictionary_path(), encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, ImportError):
        return {}


def llm_diacritizer(text: str) -> str:
    """Full tashkeel from the configured LLM (Gemini etc.). The processor rejects any answer
    that changed a letter, so a bad answer only means "no tashkeel", never altered words."""
    from app.services import llm

    prompt = ("Add full Arabic diacritics (tashkeel) to the text between the markers, for correct "
              "Modern Standard Arabic pronunciation by a text-to-speech voice. Do not add, remove, "
              "reorder or translate any word or punctuation; keep Latin words exactly. Return only the "
              "diacritized text, without the markers.\n<<<\n" + text + "\n>>>")
    answer = llm._generate_response(prompt)
    return str(answer or "").strip().strip("<>").strip()
