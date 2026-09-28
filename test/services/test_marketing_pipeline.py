"""End-to-end smoke test of the website ad pipeline without a GPU.

Real: Playwright reading a local website, the compositor, packaging, assembly,
subtitles. Faked: the LLM answer, edge-tts (a tone), and Kaggle (it returns
green-screen "talking" clips of a red presenter).
"""

import functools
import http.server
import json
import os
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from moviepy import VideoFileClip
from PIL import Image, ImageDraw

from app.config import config
from app.services.marketing import pipeline
from app.services.presenter import profiles, studio
from app.services.presenter import package as job_package
from app.utils import utils

from test.services.test_marketing_website import SERVICE_PAGE

FFMPEG = utils.get_ffmpeg_binary()


def _tts(text, voice_name, voice_rate, voice_file):
    seconds = max(1.5, min(6.5, len(text.split()) * 0.45))
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=330:duration={seconds}", voice_file], check=True)
    _tts.voices.append(voice_name)
    return object()


_tts.voices = []


def _subtitle(sub_maker, text, subtitle_file):
    with open(subtitle_file, "w", encoding="utf-8") as fp:
        fp.write(f"1\n00:00:00,000 --> 00:00:01,400\n{text[:30]}\n\n")


class FakeKaggle:
    """Renders every cloud shot: a red presenter on a green screen, the length of its narration."""

    def __init__(self):
        self.runs = 0
        self.calls = []

    def _render(self, job_id):
        self.runs += 1
        root = job_package.job_dir(job_id)
        with open(os.path.join(root, "package", "job.json"), encoding="utf-8") as fp:
            job = json.load(fp)
        os.makedirs(os.path.join(root, "output", "shots"), exist_ok=True)
        done = 0
        for shot in job["shots"]:
            if shot.get("local"):
                continue
            target = os.path.join(root, "output", "shots", f"{shot['id']}.mp4")
            if os.path.isfile(target):
                continue
            fill = "0x00FF00" if shot.get("green") else "0x3355AA"
            subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            f"color=c={fill}:s=480x832:r=25:d={shot['duration'] + 0.3}", "-vf",
                            "drawbox=x=150:y=200:w=180:h=632:color=red:t=fill", "-pix_fmt", "yuv420p", target],
                           check=True)
            done += 1
        total = sum(1 for s in job["shots"] if not s.get("local"))
        return {"total": total, "done": total, "complete": True, "rendered_now": done}

    def run_job(self, job_id, on_status=None, max_runs=4, continuing=False):
        self.calls.append("run")
        return self._render(job_id)

    def resume(self, job_id, max_runs=4, on_status=None):
        self.calls.append("resume")
        return self._render(job_id)


LLM_ANSWER = json.dumps({
    "hook": "هل بحثك جاهز؟", "message": "تدقيق أكاديمي", "cta": "اشترك الآن",
    "scenes": [
        {"type": "TALK", "duration": 5, "voiceover": "هل تريد بحثاً خالياً من الأخطاء؟",
         "presenter_action": "smiles and asks the viewer", "visual_prompt": "a modern library", "claims": ["t1"]},
        {"type": "WEBSITE_WORLD", "duration": 7, "voiceover": "نراجع بحثك لغوياً وأكاديمياً ونضمن سلامة المراجع.",
         "presenter_action": "gestures to the big screen", "website_asset": "hero", "claims": ["t3"]},
        {"type": "POINT", "duration": 6, "voiceover": "ونسلّم خلال 48 ساعة فقط.",
         "presenter_action": "points at the card", "website_asset": "card1", "claims": ["t5"]},
        {"type": "WEBSITE", "duration": 5, "voiceover": "كل ذلك من صفحة واحدة.", "website_asset": "page"},
        {"type": "CTA", "duration": 4, "voiceover": "اشترك الآن.", "website_asset": "cta1", "claims": []},
    ]})


class TestWebsiteAdPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.site = tempfile.mkdtemp()
        with open(os.path.join(cls.site, "service.html"), "w", encoding="utf-8") as fp:
            fp.write(SERVICE_PAGE)
        Image.new("RGB", (240, 80), (16, 185, 129)).save(os.path.join(cls.site, "logo.png"))
        handler = functools.partial(_Quiet, directory=cls.site)
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/service.html"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [
            mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp}),
            mock.patch.dict(config.app, {"presenters_dir": "", "video_resolution": "720p", "default_presenter": "QAI",
                                         "branding_dir": os.path.join(self.tmp, "branding")}),
            mock.patch.object(job_package.voice, "tts", _tts),
            mock.patch.object(job_package.voice, "create_subtitle", _subtitle),
        ]
        for patch in self.patches:
            patch.start()
        photo = os.path.join(self.tmp, "qai.png")
        img = Image.new("RGB", (512, 768), (200, 200, 200))
        ImageDraw.Draw(img).rectangle([180, 150, 330, 768], fill=(200, 30, 30))
        img.save(photo)
        profiles.save_presenter(profiles.Presenter(name="QAI"), [photo])
        _tts.voices.clear()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def _analyze(self, language="ar", platform="facebook"):
        return pipeline.analyze(self.url, language, 25, platform, "subscriptions",
                                generate=lambda prompt: LLM_ANSWER)

    def _wait(self, job_id):
        studio._threads[job_id].join(timeout=600)
        status = studio.read_status(job_id)
        self.assertEqual(status["state"], "done", status)
        return status

    def test_analyze_saves_project_with_default_presenter(self):
        project = self._analyze()
        self.assertEqual(project["presenter"], "QAI")
        self.assertEqual(project["aspect"], "16:9")
        self.assertEqual([s["type"] for s in project["plan"]["scenes"]],
                         ["TALK", "WEBSITE_WORLD", "POINT", "WEBSITE", "CTA"])
        self.assertEqual(pipeline.load_project(project["project_id"])["plan"]["hook"], "هل بحثك جاهز؟")
        self.assertIn(project["project_id"], pipeline.list_projects())
        self.assertEqual(self._analyze(platform="tiktok")["aspect"], "9:16")

    def test_same_page_is_read_once_then_reused(self):
        reads = []

        def counting_read(url, out_dir, locale="ar"):
            reads.append(url)
            return pipeline.website.read_website(url, out_dir, locale=locale)

        first = pipeline.analyze(self.url, "ar", 25, "facebook", "subscriptions", read=counting_read,
                                 generate=lambda prompt: LLM_ANSWER)
        second = pipeline.analyze(self.url, "ar", 40, "youtube", "leads", read=counting_read,
                                  generate=lambda prompt: LLM_ANSWER)
        self.assertEqual(len(reads), 1)  # the slow part ran once
        self.assertFalse(first["site_reused"])
        self.assertTrue(second["site_reused"])
        site = pipeline.website.load_website(second["website_dir"])
        shot = site["screenshots"][0]
        self.assertTrue(os.path.isfile(os.path.join(second["website_dir"], shot["path"])))  # images came along
        pipeline.analyze(self.url, "ar", 25, "facebook", "subscriptions", read=counting_read,
                         generate=lambda prompt: LLM_ANSWER, reuse_site=False)
        self.assertEqual(len(reads), 2)  # "read the website again"
        pipeline.analyze(self.url, "en", 25, "facebook", "subscriptions", read=counting_read,
                         generate=lambda prompt: LLM_ANSWER)
        self.assertEqual(len(reads), 3)  # another language is another reading

    def test_rebuilt_local_project_is_read_again(self):
        project = os.path.join(self.tmp, "site")
        os.makedirs(os.path.join(project, ".next"))
        build_id = os.path.join(project, ".next", "BUILD_ID")
        with open(build_id, "w") as fp:
            fp.write("a")
        key = pipeline._site_cache_key("https://qai-vo.com/x", "ar", project, "/x")
        self.assertEqual(key, pipeline._site_cache_key("https://qai-vo.com/x", "ar", project, "/x"))
        os.utime(build_id, (1, 1))
        self.assertNotEqual(key, pipeline._site_cache_key("https://qai-vo.com/x", "ar", project, "/x"))

    def test_voice_choice_goes_into_the_job_and_a_new_voice_means_a_new_render(self):
        with mock.patch.object(config, "save_config"):
            project = self._analyze()
            pid = project["project_id"]
            first = pipeline.create_job(pid, "preview")
            self.assertEqual(job_package.load_plan(first)["voice_style"], pipeline.voice_style(project))
            pipeline.set_voice_style(pid, "edge_tashkeel")
            self.assertEqual(config.app["voice_style_ar"], "edge_tashkeel")  # default for the next video
            fake = FakeKaggle()
            with mock.patch.object(studio, "start_kaggle_render") as render:
                job_id = pipeline.start(pid, "preview", "token", {}, agent=fake)
            self.assertNotEqual(job_id, first)
            self.assertEqual(job_package.load_plan(job_id)["voice_style"], "edge_tashkeel")
            render.assert_called_once()

    def test_scene_mapping(self):
        shots = [pipeline.scene_to_shot({"id": f"s{i}", "type": t, "voiceover": "x"})
                 for i, t in enumerate(["TALK", "WEBSITE_WORLD", "POINT", "CTA", "AI_SCENE", "WEBSITE"])]
        self.assertEqual([s["type"] for s in shots], ["TALK", "TALK", "TALK", "TALK", "AI_SCENE", "BROLL"])
        self.assertEqual([bool(s.get("green")) for s in shots], [False, True, True, True, False, False])
        self.assertTrue(shots[-1]["local"])

    def test_preview_then_full_video(self):
        project = self._analyze()
        agent = FakeKaggle()
        # --- 10 second preview: presenter in the website world + the real page, 720p.
        job_id = pipeline.start(project["project_id"], "preview", "token", {"bgm_type": ""}, agent=agent)
        status = self._wait(job_id)
        with VideoFileClip(status["final"]) as clip:
            self.assertEqual(clip.size, [1280, 720])
            self.assertTrue(7 <= clip.duration <= 13, clip.duration)
            first = clip.get_frame(2.0)
            last = clip.get_frame(clip.duration - 0.5)
        red = (first[..., 0] > 150) & (first[..., 1] < 90) & (first[..., 2] < 90)
        self.assertGreater(red.sum(), 2000)  # keyed presenter is there...
        self.assertGreater(np.where(red.any(axis=0))[0].mean(), 640)  # ...on the right for Arabic
        green = (first[..., 1] > 200) & (first[..., 0] < 60) & (first[..., 2] < 60)
        self.assertLess(green.sum(), 50)  # no green screen left
        button = (np.abs(last.astype(int) - (16, 185, 129)).sum(axis=2) < 60)
        self.assertGreater(button.sum(), 200)  # the real page (with its green button) fills the last scene
        job = json.load(open(os.path.join(job_package.job_dir(job_id), "package", "job.json"), encoding="utf-8"))
        self.assertEqual([(s["type"], bool(s.get("green")), bool(s.get("local"))) for s in job["shots"]],
                         [("TALK", True, False), ("BROLL", False, True)])
        self.assertTrue(all(v.startswith("ar-") for v in _tts.voices))

        # --- full video: every scene, then assembled with subtitles.
        full_id = pipeline.start(project["project_id"], "full", "token", {"bgm_type": ""}, agent=agent)
        status = self._wait(full_id)
        with VideoFileClip(status["final"]) as clip:
            self.assertEqual(clip.size, [1280, 720])
            job = json.load(open(os.path.join(job_package.job_dir(full_id), "package", "job.json"),
                                 encoding="utf-8"))
            self.assertAlmostEqual(clip.duration, sum(s["duration"] for s in job["shots"]), delta=0.3)
        self.assertEqual(pipeline.load_project(project["project_id"])["jobs"],
                         {"preview": job_id, "full": full_id})

    def test_interrupted_job_is_resumed_not_recreated(self):
        project = self._analyze(language="en", platform="instagram")
        job_id = pipeline.create_job(project["project_id"], "preview")
        studio.set_status(job_id, "running", "sending the job to Kaggle")  # the laptop was switched off here
        agent = FakeKaggle()
        again = pipeline.start(project["project_id"], "preview", "token", {"bgm_type": ""}, agent=agent)
        self.assertEqual(again, job_id)
        status = self._wait(job_id)
        with VideoFileClip(status["final"]) as clip:
            self.assertEqual(clip.size, [720, 1280])
        self.assertTrue(all(v.startswith("en-") for v in _tts.voices))

    def test_failed_kaggle_run_is_rendered_again_not_resumed(self):
        """Regression: after a Kaggle run failed (model download broke), pressing Preview only
        re-read the old failed run ('failed before making any shot') instead of trying again."""
        project = self._analyze()
        job_id = pipeline.create_job(project["project_id"], "preview")
        studio.prepare_package(job_id)
        with open(os.path.join(job_package.job_dir(job_id), "kaggle.json"), "w", encoding="utf-8") as fp:
            json.dump({"kernel": "maya/mpt-presenter-x", "dataset": "maya/mpt-job-x"}, fp)
        studio.set_status(job_id, "error", "ERROR: The Kaggle run failed before making any shot.", kaggle="error")
        self.assertFalse(studio.can_continue(job_id))
        agent = FakeKaggle()
        self.assertEqual(pipeline.start(project["project_id"], "preview", "token", {"bgm_type": ""}, agent=agent),
                         job_id)  # same job, a new run
        self._wait(job_id)
        self.assertEqual(agent.calls, ["run"])

    def test_laptop_error_after_finished_kaggle_run_only_fetches(self):
        project = self._analyze()
        job_id = pipeline.create_job(project["project_id"], "preview")
        studio.prepare_package(job_id)
        with open(os.path.join(job_package.job_dir(job_id), "kaggle.json"), "w", encoding="utf-8") as fp:
            json.dump({"kernel": "maya/mpt-presenter-x", "dataset": "maya/mpt-job-x"}, fp)
        studio.set_status(job_id, "error", "ERROR: 'charmap' codec can't encode", kaggle="complete")
        self.assertTrue(studio.can_continue(job_id))
        agent = FakeKaggle()
        pipeline.start(project["project_id"], "preview", "token", {"bgm_type": ""}, agent=agent)
        self._wait(job_id)
        self.assertEqual(agent.calls, ["resume"])  # no new GPU run

    def test_missing_presenter_render_falls_back_to_photo(self):
        project = self._analyze()
        job_id = pipeline.create_job(project["project_id"], "preview")
        studio.prepare_package(job_id)
        final = pipeline.render(job_id, {"bgm_type": ""})  # nothing came back from Kaggle
        with VideoFileClip(final) as clip:
            frame = clip.get_frame(2.0)
        self.assertGreater(((frame[..., 0] > 150) & (frame[..., 1] < 90)).sum(), 1000)

    def test_storyboard_edits_are_validated(self):
        project = self._analyze()
        scenes = project["plan"]["scenes"]
        scenes[1]["website_asset"] = "made-up"
        scenes[2]["voiceover"] = "خصم 90% لأول 1000 مشترك"
        updated = pipeline.update_scenes(project["project_id"], scenes)["plan"]
        self.assertEqual(updated["scenes"][1]["website_asset"], "card1")
        self.assertTrue(updated["scenes"][2]["needs_review"])


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
