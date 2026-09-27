import os
import subprocess
import tempfile
import unicodedata
import unittest

import numpy as np
from PIL import ImageFont

from app.models.schema import VideoParams
from app.services import video
from app.utils import rtl_text, utils

TAJAWAL = os.path.join(utils.font_dir(), rtl_text.DEFAULT_ARABIC_FONT)
STHEITI = os.path.join(utils.font_dir(), "STHeitiMedium.ttc")
ARABIC_ALPHABET = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئةى"


def _logical_order(visual_line: str) -> str:
    """Undo the visual reordering so tests can compare against the script."""
    return visual_line[::-1]


def _draws_glyph(font, char, missing_signature) -> bool:
    mask = font.getmask(char)
    return mask.getbbox() is not None and (mask.size, bytes(mask)) != missing_signature


class TestRtlDetection(unittest.TestCase):
    def test_contains_rtl(self):
        self.assertTrue(rtl_text.contains_rtl("مرحبا"))
        self.assertTrue(rtl_text.contains_rtl("Hello مرحبا"))
        self.assertFalse(rtl_text.contains_rtl("Hello 123"))
        self.assertFalse(rtl_text.contains_rtl(""))
        self.assertFalse(rtl_text.contains_rtl(None))

    def test_is_rtl_language(self):
        self.assertTrue(rtl_text.is_rtl_language("ar"))
        self.assertTrue(rtl_text.is_rtl_language("ar-SA"))
        self.assertTrue(rtl_text.is_rtl_language("fa_IR"))
        self.assertFalse(rtl_text.is_rtl_language("en-US"))
        self.assertFalse(rtl_text.is_rtl_language(""))


class TestShaping(unittest.TestCase):
    def test_latin_text_is_unchanged(self):
        text = "Hello, world! 2026"
        self.assertEqual(rtl_text.shape_line(text), text)
        self.assertEqual(rtl_text.shape_text("a\nb"), "a\nb")

    def test_letters_are_joined_and_reversed(self):
        # "بيت": beh initial + yeh medial + teh final, drawn right-to-left.
        shaped = rtl_text.shape_line("بيت")
        self.assertEqual(shaped, "ﺖﻴﺑ")

    def test_isolated_forms_use_base_letters(self):
        shaped = rtl_text.shape_line("و ب ا")
        for char in shaped:
            self.assertFalse(
                unicodedata.decomposition(char).startswith("<isolated>"),
                f"isolated presentation form left in output: {hex(ord(char))}",
            )
        self.assertEqual(shaped, "ا ب و")

    def test_lam_alef_ligature(self):
        shaped = rtl_text.shape_line("لا")
        self.assertEqual(len(shaped), 1)
        self.assertEqual(unicodedata.name(shaped), "ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM")

    def test_embedded_latin_and_numbers_stay_left_to_right(self):
        shaped = rtl_text.shape_line("تطبيق ChatGPT الجديد عام 2026")
        self.assertIn("ChatGPT", shaped)
        self.assertIn("2026", shaped)
        # The first Arabic word is drawn at the right end of the line.
        self.assertTrue(shaped.endswith(rtl_text.shape_line("تطبيق")))

    def test_multiline_keeps_line_order(self):
        shaped = rtl_text.shape_text("واحد\nاثنان")
        first, second = shaped.split("\n")
        self.assertEqual(first, rtl_text.shape_line("واحد"))
        self.assertEqual(second, rtl_text.shape_line("اثنان"))

    def test_bundled_font_draws_every_shaped_letter(self):
        font = ImageFont.truetype(TAJAWAL, 40, layout_engine=ImageFont.Layout.BASIC)
        missing = font.getmask("\U0010ffff")
        missing_signature = (missing.size, bytes(missing))
        # Every letter in initial, medial, final and isolated position.
        words = [f"{c}" for c in ARABIC_ALPHABET]
        words += [f"{c}ب" for c in ARABIC_ALPHABET]
        words += [f"ب{c}ب" for c in ARABIC_ALPHABET]
        words += [f"ب{c}" for c in ARABIC_ALPHABET]
        words += ["لا", "بلا", "لأن", "إلا", "آلام", "الله"]
        shaped = rtl_text.shape_line(" ".join(words))
        missing_glyphs = sorted(
            {
                f"{hex(ord(c))} {unicodedata.name(c, '?')}"
                for c in shaped
                if not c.isspace() and not _draws_glyph(font, c, missing_signature)
            }
        )
        self.assertEqual(missing_glyphs, [])


class TestBasicLayout(unittest.TestCase):
    def test_basic_layout_is_scoped(self):
        original = ImageFont.truetype
        with rtl_text.basic_layout():
            font = ImageFont.truetype(TAJAWAL, 20)
            self.assertEqual(font.layout_engine, ImageFont.Layout.BASIC)
        self.assertIs(ImageFont.truetype, original)

    def test_basic_layout_restores_on_error(self):
        original = ImageFont.truetype
        with self.assertRaises(RuntimeError):
            with rtl_text.basic_layout():
                raise RuntimeError("boom")
        self.assertIs(ImageFont.truetype, original)


class TestFontFallback(unittest.TestCase):
    def test_font_detection(self):
        self.assertTrue(rtl_text.font_has_arabic(TAJAWAL))
        self.assertFalse(rtl_text.font_has_arabic(STHEITI))

    def test_resolve_font_for_arabic_text(self):
        resolved = rtl_text.resolve_font_for_text(STHEITI, "مرحبا", utils.font_dir())
        self.assertTrue(resolved.replace("\\", "/").endswith("Tajawal-Bold.ttf"))

    def test_resolve_font_keeps_font_for_latin_text(self):
        self.assertEqual(
            rtl_text.resolve_font_for_text(STHEITI, "hello", utils.font_dir()), STHEITI
        )


class TestArabicWrapping(unittest.TestCase):
    TEXT = (
        "مرحباً بكم في هذا الفيديو عن الذكاء الاصطناعي، وكيف يغير حياتنا "
        "اليومية في المدرسة والعمل والبيت خلال السنوات القادمة"
    )

    def test_wrapped_lines_fit_and_keep_reading_order(self):
        max_width = 700
        wrapped, height = video.wrap_text(
            self.TEXT, max_width=max_width, font=TAJAWAL, fontsize=60
        )
        lines = wrapped.split("\n")
        self.assertGreater(len(lines), 1)
        font = ImageFont.truetype(TAJAWAL, 60, layout_engine=ImageFont.Layout.BASIC)
        for line in lines:
            left, _, right, _ = font.getbbox(line)
            self.assertLessEqual(right - left, max_width)
        # The first line holds the *first* words of the sentence.
        self.assertTrue(lines[0].endswith(rtl_text.shape_line("مرحباً")))
        self.assertTrue(lines[-1].startswith(rtl_text.shape_line("القادمة")))
        # No words lost or duplicated by wrapping.
        words = [w for line in lines for w in line.split(" ")]
        self.assertEqual(len(words), len(self.TEXT.split(" ")))
        ascent, descent = font.getmetrics()
        self.assertEqual(height, len(lines) * (ascent + descent))

    def test_arabic_comma_never_starts_a_line(self):
        text = "كلمة " * 12 + "، " + "كلمة " * 12
        wrapped, _ = video.wrap_text(text.strip(), max_width=400, font=TAJAWAL, fontsize=60)
        for line in wrapped.split("\n"):
            # Visual order: the logical first character is the rightmost one.
            self.assertNotEqual(line.rstrip()[-1:], "،")

    def test_short_arabic_text_is_shaped_without_wrapping(self):
        wrapped, _ = video.wrap_text("مرحبا", max_width=1000, font=TAJAWAL, fontsize=60)
        self.assertEqual(wrapped, rtl_text.shape_line("مرحبا"))


class TestArabicSubtitleRender(unittest.TestCase):
    """Render a real frame through generate_video and inspect its pixels."""

    def test_arabic_subtitle_frame_is_rendered(self):
        ffmpeg = utils.get_ffmpeg_binary()
        with tempfile.TemporaryDirectory() as tmp:
            bg = os.path.join(tmp, "bg.mp4")
            audio = os.path.join(tmp, "a.mp3")
            srt = os.path.join(tmp, "s.srt")
            out = os.path.join(tmp, "out.mp4")
            frame = os.path.join(tmp, "frame.png")
            subprocess.run(
                [ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                 "color=c=black:s=1080x1920:d=2", "-r", "30", bg],
                check=True,
            )
            subprocess.run(
                [ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                 "anullsrc=r=44100:cl=mono", "-t", "2", audio],
                check=True,
            )
            with open(srt, "w", encoding="utf-8") as f:
                f.write("1\n00:00:00,000 --> 00:00:02,000\nلا إله إلا الله\n\n")
            params = VideoParams(
                video_subject="t",
                video_aspect="9:16",
                font_name="STHeitiMedium.ttc",  # no Arabic glyphs -> fallback
                font_size=80,
                text_fore_color="#FFFFFF",
                stroke_width=0,
                subtitle_position="center",
                text_background_color=False,
                bgm_type="",
                n_threads=1,
            )
            video.generate_video(bg, audio, srt, out, params)
            subprocess.run(
                [ffmpeg, "-loglevel", "error", "-y", "-ss", "1", "-i", out,
                 "-frames:v", "1", frame],
                check=True,
            )
            from PIL import Image

            pixels = np.asarray(Image.open(frame).convert("L"))
            bright = np.argwhere(pixels > 128)
            self.assertGreater(len(bright), 500, "subtitle text was not drawn")
            # Text is horizontally centred around the middle of the frame.
            centre_x = (bright[:, 1].min() + bright[:, 1].max()) / 2
            self.assertAlmostEqual(centre_x, 540, delta=60)


if __name__ == "__main__":
    unittest.main()
