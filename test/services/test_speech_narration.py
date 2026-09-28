import datetime
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from app.config import config
from app.services.presenter import package as job_package
from app.services.speech import narration
from app.utils import utils


def edge_like(text):
    """A SubMaker as edge-tts 7 returns it: one timed cue per spoken word, 0.4 s each."""
    words = text.split()
    cues = [SimpleNamespace(content=w, start=datetime.timedelta(seconds=0.4 * i),
                            end=datetime.timedelta(seconds=0.4 * i + 0.35)) for i, w in enumerate(words)]
    return SimpleNamespace(cues=cues)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()

    def tearDown(self):
        self.env.stop()


class TestNarration(Case):
    def test_voice_reads_spoken_form_subtitles_keep_display(self):
        display = "مرحباً بكم في QAI-VO. نتحقق من روابط DOI بسرعة."
        spoken_seen = []

        def fake_tts(text, voice_name, voice_rate, voice_file):
            spoken_seen.append(text)
            subprocess.run([utils.get_ffmpeg_binary(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            "sine=frequency=300:duration=4", voice_file], check=True)
            return edge_like(text)

        shot = {"id": "s01", "narration": display}
        audio = os.path.join(self.tmp, "s01.mp3")
        srt = os.path.join(self.tmp, "s01.srt")
        job_package.record_narration(shot, audio, srt, "ar-SA-ZariyahNeural-Female", 1.0, tts=fake_tts)
        self.assertIn("كيو إيه آي ڤي أو", spoken_seen[0])
        self.assertIn("دي أو آي", spoken_seen[0])
        self.assertNotIn("QAI-VO", spoken_seen[0])
        self.assertEqual(shot["spoken"], spoken_seen[0])
        with open(srt, encoding="utf-8") as fp:
            subtitles = fp.read()
        self.assertIn("QAI-VO", subtitles)  # display text on screen
        self.assertNotIn("ڤي", subtitles)
        self.assertIn("DOI", subtitles)
        blocks = [b for b in subtitles.strip().split("\n\n") if b]
        self.assertEqual(len(blocks), 2)
        # The second line starts after the first one's spoken words: "مرحباً بكم في كيو إيه آي ڤي أو"
        # is 8 words x 0.4 s, even though the displayed line has only 4 words.
        self.assertIn("00:00:03,200 -->", blocks[1])

    def test_plain_text_keeps_the_old_path(self):
        seen = []

        def fake_tts(text, voice_name, voice_rate, voice_file):
            seen.append(text)
            open(voice_file, "wb").write(b"x")
            return edge_like(text)

        shot = {"id": "s02", "narration": "نص عربي عادي بدون أسماء."}
        with mock.patch("app.services.voice.create_subtitle") as create, \
                mock.patch("app.services.voice.get_audio_duration", return_value=2.0):
            job_package.record_narration(shot, os.path.join(self.tmp, "a.mp3"), os.path.join(self.tmp, "a.srt"),
                                         "ar-SA-ZariyahNeural-Female", 1.0, tts=fake_tts)
        self.assertEqual(seen, ["نص عربي عادي بدون أسماء."])
        create.assert_called_once()

    def test_english_voice_spells_letters(self):
        processor = narration.processor_for("en-US-JennyNeural")
        self.assertEqual(narration.prepare("Try QAI-VO today", processor)["spoken"], "Try Q A I V O today.")

    def test_edge_and_silma_keep_digits_other_engines_get_words(self):
        edge = narration.prepare("خلال 48 ساعة", narration.processor_for("ar-SA-X"))
        silma = narration.prepare("خلال 48 ساعة", narration.processor_for("ar-SA-X", engine="silma"))
        other = narration.prepare("خلال 48 ساعة", narration.processor_for("ar-SA-X", engine="f5"))
        self.assertIn("48", edge["spoken"])
        self.assertIn("48", silma["spoken"])  # SILMA's own normaliser reads it
        self.assertIn("ثمانية وأربعون", other["spoken"])

    def test_tashkeel_is_opt_in(self):
        with mock.patch("app.services.speech.narration.llm_diacritizer", return_value="نَصٌّ") as llm:
            narration.prepare("نص", narration.processor_for("ar-SA-X"))
            llm.assert_not_called()
            with mock.patch.dict(config.app, {"pronunciation_tashkeel": True}):
                result = narration.prepare("نص", narration.processor_for("ar-SA-X"))
        self.assertEqual(result["spoken"], "نَصٌّ.")

    def test_legacy_offsets(self):
        maker = SimpleNamespace(cues=None, offset=[(0, 5_000_000), (5_000_000, 10_000_000)], subs=["a", "b"])
        self.assertEqual(narration.word_times(maker), [(0.0, 0.5), (0.5, 1.0)])


if __name__ == "__main__":
    unittest.main()
