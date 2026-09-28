"""Voice Lab: every setup side by side, with the SILMA path going through the real RunPod handler
(fake RunPod API + a stand-in SILMA script; no network, no GPU)."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from app.config import config
from app.services.speech import voice_lab
from app.utils import utils
from test.services.test_presenter_runpod import FakeRunPod, load_handler

FFMPEG = utils.get_ffmpeg_binary()

FAKE_SILMA = r'''
import json, os, subprocess, sys
request, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
items = json.load(open(request, encoding="utf-8"))["items"]
results = []
for item in items:
    assert item["text"]
    if item.get("ref_wav"):
        assert os.path.getsize(item["ref_wav"]) > 1000 and item["ref_text"]
    path = os.path.join(out, item["id"] + ".wav")
    subprocess.run([sys.argv[3], "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                    "-ar", "24000", "-ac", "1", path], check=True)
    results.append({"id": item["id"], "file": os.path.basename(path), "sample_rate": 24000, "seconds": 3.0,
                    "inference_seconds": 0.4, "used_ref": bool(item.get("ref_wav"))})
json.dump({"engine": "silma", "model_load_seconds": 12.5, "items": results}, open(os.path.join(out, "result.json"), "w"))
'''


def fake_edge(text, voice_name, voice_rate, voice_file):
    fake_edge.calls.append((text, voice_name))
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
                    voice_file], check=True)
    return object()


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()
        fake_edge.calls = []

    def tearDown(self):
        self.env.stop()

    def silma_agent(self):
        from app.services.presenter import runpod_agent

        handler = load_handler()
        handler.VOLUME = os.path.join(self.tmp, "volume")
        os.makedirs(handler.VOLUME)
        script = os.path.join(self.tmp, "fake_silma.py")
        with open(script, "w", encoding="utf-8") as fp:
            fp.write(FAKE_SILMA)
        wrapper = os.path.join(self.tmp, "silma_python")
        with open(wrapper, "w", encoding="utf-8") as fp:
            fp.write(f"#!/bin/sh\nexec {sys.executable} {script} \"$2\" \"$3\" {FFMPEG}\n")
        os.chmod(wrapper, 0o755)
        handler.SILMA_PYTHON = wrapper
        fake = FakeRunPod(handler)
        return runpod_agent.RunPodAgent(api_key="key", endpoint_id="ep", session=fake, poll_seconds=0,
                                        log=lambda m: None, sleep=lambda s: None), fake


class TestVoiceLab(Case):
    def test_all_setups_side_by_side_with_metadata(self):
        agent, fake = self.silma_agent()
        with mock.patch("app.services.speech.voice_lab.llm_diacritizer", side_effect=lambda t: t):
            report = voice_lab.run(list(voice_lab.ENGINES), ["ar", "en"], tts=fake_edge, agent=agent)
        names = sorted((r["engine"], r["language"]) for r in report["results"])
        self.assertEqual(len(names), 9)  # tashkeel only for Arabic
        self.assertNotIn(("edge_tashkeel", "en"), names)
        by = {(r["engine"], r["language"]): r for r in report["results"]}
        for entry in report["results"]:
            self.assertEqual(entry["error"], "", entry)
            self.assertTrue(os.path.isfile(os.path.join(report["folder"], entry["file"])))
            self.assertGreater(entry["duration"], 1)
            self.assertIn("loudness_lufs", entry)
        # Edge as today reads the text as written; the fixed setups read QAI-VO letter by letter.
        self.assertIn("QAI-VO", by[("edge", "ar")]["spoken_text"])
        self.assertIn("كيو إيه آي ڤي أو", by[("edge_fixed", "ar")]["spoken_text"])
        self.assertIn("Q A I V O", by[("silma", "en")]["spoken_text"])
        self.assertIn("48", by[("silma", "ar")]["spoken_text"])  # SILMA reads numbers itself
        # SILMA: one RunPod call for all its samples; Lina's voice sent as the reference.
        self.assertEqual(fake.modes(), ["tts"])
        self.assertEqual(by[("silma_lina", "ar")]["model_load_seconds"], 12.5)
        self.assertTrue(os.path.isfile(os.path.join(report["folder"], "reference_ar.wav")))
        with open(os.path.join(report["folder"], "benchmark.json"), encoding="utf-8") as fp:
            saved = json.load(fp)
        self.assertEqual(len(saved["results"]), 9)
        self.assertTrue(report["folder"].startswith(self.tmp))  # data drive, not C:

    def test_silma_failure_is_reported_not_hidden(self):
        from app.services.presenter import runpod_agent

        broken = mock.Mock()
        broken.call.side_effect = runpod_agent.RunPodError("RunPod endpoint not found")
        report = voice_lab.run(["edge", "silma"], ["ar"], tts=fake_edge, agent=broken)
        by = {r["engine"]: r for r in report["results"]}
        self.assertEqual(by["edge"]["error"], "")
        self.assertIn("endpoint not found", by["silma"]["error"])
        self.assertEqual(by["silma"]["file"], "")

    def test_background_run_and_listing(self):
        folder = voice_lab.start(["edge"], ["en"], tts=fake_edge)
        for _ in range(100):
            status, report = voice_lab.read(folder)
            if status.get("state") != "running":
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "done")
        self.assertEqual(voice_lab.runs()[0], folder)
        self.assertEqual(report["results"][0]["engine"], "edge")


class TestBrandByEar(Case):
    def test_versions_made_and_choice_saved(self):
        with mock.patch.object(config, "save_config"), mock.patch.dict(config.app, {}):
            made = voice_lab.brand_test("ar", tts=fake_edge)
            self.assertEqual(len(made), len(voice_lab.saved_brand_test("ar")))
            self.assertIn(made[1]["variant"], fake_edge.calls[1][0])  # each version read in the sentence
            voice_lab.choose_brand("ar", made[1]["variant"])
            self.assertEqual(config.app["qai_vo_spoken_ar"], made[1]["variant"])


class TestVoiceLabPage(Case):
    def test_page_without_runpod_keeps_edge_only(self):
        from streamlit.testing.v1 import AppTest

        page = os.path.join(os.path.dirname(__file__), "..", "..", "webui", "pages", "4_Voice_Lab.py")
        with mock.patch.dict(config.app, {"runpod_api_key": "", "runpod_endpoint_id": ""}), \
                mock.patch.object(config, "save_config"):
            app = AppTest.from_file(page, default_timeout=60)
            app.run()
        self.assertFalse(app.exception, app.exception)
        self.assertIn("SILMA بيشتغل على RunPod: جهّزي RunPod بصفحة Presenter Video.", [i.value for i in app.info])


if __name__ == "__main__":
    unittest.main()
