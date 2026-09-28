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


class TestVoiceStyles(Case):
    def _tts(self, seen):
        def tts(text, voice_name, voice_rate, voice_file):
            seen.append(text)
            subprocess.run([utils.get_ffmpeg_binary(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            "sine=frequency=300:duration=2", voice_file], check=True)
            return edge_like(text)
        return tts

    def test_each_edge_style_reads_the_right_text(self):
        display = "مرحباً بكم في QAI-VO."
        results = {}
        for style in ("edge_plain", "edge_fixed", "edge_tashkeel"):
            seen = []
            with mock.patch("app.services.speech.narration.llm_diacritizer", side_effect=lambda t: t.replace("بكم", "بِكُمْ")):
                job_package.record_narration({"id": "s1", "narration": display}, os.path.join(self.tmp, f"{style}.mp3"),
                                             os.path.join(self.tmp, f"{style}.srt"), "ar-SA-ZariyahNeural-Female", 1.0,
                                             tts=self._tts(seen), style=style)
            results[style] = seen[0]
        self.assertIn("QAI-VO", results["edge_plain"])  # as written
        self.assertIn("كيو إيه آي", results["edge_fixed"])
        self.assertNotIn("بِكُمْ", results["edge_fixed"])
        self.assertIn("بِكُمْ", results["edge_tashkeel"])

    def test_silma_style_records_all_shots_in_one_runpod_call(self):
        from app.services.presenter import profiles
        from PIL import Image

        photo = os.path.join(self.tmp, "p.png")
        Image.new("RGB", (400, 600), (180, 160, 150)).save(photo)
        with mock.patch.dict(config.app, {"presenters_dir": ""}):
            presenter = profiles.save_presenter(profiles.Presenter(name="Lina"), [photo])
        presenter.voice_name = "ar-SA-ZariyahNeural-Female"
        calls = []

        class FakeAgent:
            def call(self, payload, timeout=0):
                import base64

                calls.append(payload)
                items = []
                for item in payload["items"]:
                    path = os.path.join(self_tmp, item["id"] + ".wav")
                    subprocess.run([utils.get_ffmpeg_binary(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                                    "sine=frequency=220:duration=3", path], check=True)
                    items.append({"id": item["id"], "wav": base64.b64encode(open(path, "rb").read()).decode()})
                return {"items": items}

        self_tmp = self.tmp
        shots = [{"id": "s01", "type": "TALK", "narration": "مرحباً بكم في QAI-VO. ابدأ اليوم."},
                 {"id": "s02", "type": "TALK", "narration": "نتحقق من روابط DOI."}]
        seen = []
        folder = job_package.build_package("job1", presenter, shots, {"aspect": "16:9"}, tts=self._tts(seen),
                                           voice_style="silma_lina", agent=FakeAgent())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["mode"], "tts")
        self.assertTrue(all(item.get("ref_wav") for item in calls[0]["items"]))  # Lina's voice to copy
        self.assertIn("كيو إيه آي", calls[0]["items"][0]["text"])
        self.assertEqual(len(seen), 1)  # Edge only read the short voice reference
        self.assertTrue(os.path.isfile(os.path.join(folder, "audio", "s01.mp3")))
        with open(os.path.join(job_package.job_dir("job1"), "subtitles", "s01.srt"), encoding="utf-8") as fp:
            srt = fp.read()
        self.assertIn("QAI-VO", srt)  # subtitles keep the display text
        self.assertEqual(srt.count("-->"), 2)

    def test_style_is_remembered_per_language(self):
        with mock.patch.object(config, "save_config"), mock.patch.dict(config.app, {}):
            self.assertEqual(narration.default_style("ar"), narration.DEFAULT_STYLE)
            narration.remember_style("ar", "edge_tashkeel")
            narration.remember_style("en", "edge_plain")
            self.assertEqual(narration.default_style("ar-SA"), "edge_tashkeel")
            self.assertEqual(narration.default_style("en"), "edge_plain")
