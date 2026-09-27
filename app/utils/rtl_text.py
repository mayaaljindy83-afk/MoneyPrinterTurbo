"""Right-to-left (Arabic, Persian, Urdu, Hebrew) text helpers for subtitles.

Pillow can only shape Arabic correctly when it was built with libraqm *and*
FriBiDi is available at runtime. Linux wheels usually have both, but the
official Windows wheels do not ship FriBiDi, so the same code renders
connected letters on Linux and broken, left-to-right letters on Windows.

To get identical output on every platform we never rely on raqm for RTL text:

1. ``arabic_reshaper`` picks the correct contextual glyph for every letter
   (initial / medial / final / isolated, plus lam-alef ligatures);
2. ``python-bidi`` reorders the line into visual order (RTL runs reversed,
   embedded English words and numbers kept left-to-right);
3. the result is drawn with Pillow's BASIC layout engine, which draws the
   characters exactly in the order we give them.

Line wrapping must happen on the *logical* text (before step 2), otherwise
the first words of a sentence would end up on the last line.
"""

from __future__ import annotations

import os
import unicodedata
from contextlib import contextmanager
from functools import lru_cache

from PIL import ImageFont

try:  # Optional at import time so the rest of the app keeps working.
    import arabic_reshaper
    from bidi import get_display
except ImportError:  # pragma: no cover - exercised only without the deps
    arabic_reshaper = None
    get_display = None

# Bundled font that covers Arabic and Latin. Used when the configured
# subtitle font has no Arabic glyphs (e.g. the upstream default STHeiti).
DEFAULT_ARABIC_FONT = "Tajawal-Bold.ttf"

_RTL_BIDI_CLASSES = {"R", "AL"}

# Punctuation that must never start a wrapped line in Arabic text.
ARABIC_LINE_START_PUNCTUATION = "،؛؟"

_reshaper = None


def _get_reshaper():
    global _reshaper
    if _reshaper is None and arabic_reshaper is not None:
        # Harakat (short vowel marks) are dropped: Pillow's BASIC engine cannot
        # position combining marks, and subtitles are normally unvocalised.
        _reshaper = arabic_reshaper.ArabicReshaper(
            configuration={"delete_harakat": True, "support_ligatures": True}
        )
    return _reshaper


def is_rtl_char(char: str) -> bool:
    return unicodedata.bidirectional(char) in _RTL_BIDI_CLASSES


def contains_rtl(text: str | None) -> bool:
    """Return True when the text has at least one right-to-left letter."""
    return any(is_rtl_char(char) for char in str(text or ""))


def is_rtl_language(language: str | None) -> bool:
    code = str(language or "").strip().lower().replace("_", "-")
    return code.split("-")[0] in {"ar", "fa", "ur", "he", "ps", "ku", "sd", "ug", "yi"}


@lru_cache(maxsize=1)
def _isolated_form_map() -> dict[str, str]:
    """Map every Arabic *isolated* presentation form to its base letter.

    Many modern Arabic fonts (Tajawal included) only contain the
    initial/medial/final presentation forms and rely on the base letter for
    the isolated shape. The isolated shape *is* the base glyph, so mapping it
    back is always visually identical and avoids tofu boxes.
    """
    mapping = {}
    for code_point in list(range(0xFB50, 0xFE00)) + list(range(0xFE70, 0xFF00)):
        char = chr(code_point)
        decomposition = unicodedata.decomposition(char)
        if not decomposition.startswith("<isolated>"):
            continue
        parts = decomposition.split()[1:]
        if len(parts) == 1:
            mapping[char] = chr(int(parts[0], 16))
    return mapping


def shape_line(line: str) -> str:
    """Convert one logical line into visual order with joined Arabic letters.

    Text without RTL characters is returned unchanged, so it is safe to call
    for any language.
    """
    if not contains_rtl(line):
        return line
    reshaper = _get_reshaper()
    if reshaper is None or get_display is None:
        return line
    reshaped = reshaper.reshape(line)
    isolated = _isolated_form_map()
    reshaped = "".join(isolated.get(char, char) for char in reshaped)
    return get_display(reshaped)


def shape_text(text: str) -> str:
    """Shape every line of a (possibly multi-line) string independently."""
    return "\n".join(shape_line(line) for line in str(text or "").split("\n"))


@contextmanager
def basic_layout():
    """Force Pillow's BASIC layout engine for fonts opened inside the block.

    MoviePy's ``TextClip`` opens fonts with ``ImageFont.truetype(path, size)``
    and gives us no way to pass ``layout_engine``. Already-shaped RTL text must
    not be shaped/reordered a second time by raqm, so we temporarily default
    ``layout_engine`` to BASIC. Explicit callers are left untouched.
    """
    original = ImageFont.truetype

    def truetype(font=None, size=10, index=0, encoding="", layout_engine=None):
        if layout_engine is None:
            layout_engine = ImageFont.Layout.BASIC
        return original(font, size, index, encoding, layout_engine)

    ImageFont.truetype = truetype
    try:
        yield
    finally:
        ImageFont.truetype = original


@lru_cache(maxsize=32)
def font_has_arabic(font_path: str) -> bool:
    """Best-effort check that a font can draw basic Arabic letters."""
    try:
        font = ImageFont.truetype(font_path, 30, layout_engine=ImageFont.Layout.BASIC)
        missing = font.getmask("\U0010ffff")
        missing_signature = (missing.size, bytes(missing))
        for char in "العربيﺑ":  # الغربي + medial beh
            mask = font.getmask(char)
            if mask.getbbox() is None or (mask.size, bytes(mask)) == missing_signature:
                return False
        return True
    except Exception:
        return False


def resolve_font_for_text(font_path: str, text: str, font_dir: str) -> str:
    """Return ``font_path`` or the bundled Arabic font when it cannot draw RTL text."""
    if not contains_rtl(text) or font_has_arabic(font_path):
        return font_path
    fallback = os.path.join(font_dir, DEFAULT_ARABIC_FONT)
    if os.path.isfile(fallback):
        return fallback.replace("\\", "/") if os.name == "nt" else fallback
    return font_path
