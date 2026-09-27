import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

from moviepy import AudioFileClip, VideoFileClip
from PIL import Image

from app.config import config
from app.models.schema import VideoParams
from app.services.presenter import assemble, kaggle_agent, planner, profiles
from app.services.presenter import package as job_package
from app.utils import utils

FFMPEG = utils.get_ffmpeg_binary()

ARABIC = ("أهلاً بكم في جولة سريعة على موقعنا. نحن نساعدك في كتابة الأبحاث بسرعة. "
          "يمكنك رفع ملفاتك بسهولة، ثم طرح أسئلتك على المساعد الذكي. "
          "كل النتائج تظهر خلال ثوانٍ قليلة، مع المراجع الكاملة. "
          "جرّب الآن مجاناً وشاركنا رأيك. شكراً لمتابعتكم!")


def _tone(path, seconds, freq=300):
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:duration={seconds}", path], check=True)


class StorageCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()
        self.cfg = mock.patch.dict(config.app, {"presenters_dir": ""})
        self.cfg.start()

    def tearDown(self):
        self.cfg.stop()
        self.env.stop()

    def _presenter(self):
        src = os.path.join(self.tmp, "face.jpg")
        Image.new("RGB", (600, 900), (190, 160, 150)).save(src)
        return profiles.save_presenter(profiles.Presenter(name="Lina"), [src])


class TestProfiles(StorageCase):
    def test_save_load_list(self):
        presenter = self._presenter()
        self.assertEqual(profiles.list_presenters(), ["Lina"])
        self.assertEqual([os.path.basename(p) for p in presenter.reference_images], ["ref1.png"])
        loaded = profiles.load_presenter("Lina")
        self.assertEqual(loaded.voice_name, profiles.DEFAULT_VOICE)
        self.assertIn("25-year-old woman", loaded.description)
        profiles.delete_presenter("Lina")
        self.assertEqual(profiles.list_presenters(), [])

    def test_names_cannot_escape_the_folder(self):
        self.assertEqual(profiles.safe_name("../../evil"), "evil")
        with self.assertRaises(ValueError):
            profiles.safe_name("../")


class TestPlanner(unittest.TestCase):
    def test_chunks_keep_all_text_in_order(self):
        chunks = planner.chunk_narration(ARABIC, "ar-SA")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks).split(), ARABIC.split())
        for chunk in chunks:
            self.assertLessEqual(planner.estimate_seconds(chunk, "ar"), planner.MAX_SECONDS + 4)

    def test_llm_looks_are_applied_and_rules_enforced(self):
        answer = json.dumps([
            {"type": "WALK", "location": "a library", "action": "walks", "camera": "wide"},
            {"type": "POINT", "location": "Home page screen", "action": "points at it", "camera": "medium"},
            {"type": "BROLL", "location": "a busy street", "action": "x", "camera": "pan"},
            {"type": "BROLL", "location": "the home page", "action": "", "camera": "slow pan"},
            {"type": "DANCE", "location": "", "action": "", "camera": ""},
        ])
        script = " ".join(f"Sentence number {i} is long enough to be spoken for a while here." for i in range(25))
        shots = planner.plan_shots(script, "topic", "en", places=["Home page"],
                                   generate=lambda prompt: f"Sure!\n```json\n{answer}\n```")
        self.assertEqual(shots[0]["type"], "TALK")  # first shot is always TALK
        self.assertEqual(shots[1]["type"], "POINT")
        self.assertEqual(shots[1]["place"], "Home page")
        self.assertEqual(shots[2]["type"], "WALK")  # BROLL without a photo becomes WALK
        self.assertEqual(shots[3]["type"], "BROLL")
        self.assertEqual(shots[3]["place"], "Home page")
        self.assertEqual(shots[-1]["type"], "TALK")
        self.assertEqual([s["id"] for s in shots[:3]], ["s01", "s02", "s03"])

    def test_llm_failure_uses_default_rhythm(self):
        def broken(prompt):
            raise RuntimeError("ollama is not running")

        shots = planner.plan_shots(ARABIC * 3, "topic", "ar", generate=broken)
        self.assertTrue(all(s["type"] in planner.SHOT_TYPES for s in shots))
        self.assertIn("WALK", [s["type"] for s in shots])
        self.assertEqual(planner.parse_looks("no json here", 3), [])


def _fake_tts(text, voice_name, voice_rate, voice_file):
    _tone(voice_file, 2.0)
    return object()


def _fake_subtitle(sub_maker, text, subtitle_file):
    with open(subtitle_file, "w", encoding="utf-8") as fp:
        fp.write(f"1\n00:00:00,000 --> 00:00:01,900\n{text[:20]}\n\n")


class TestPackage(StorageCase):
    def _build(self):
        presenter = self._presenter()
        shot_img = os.path.join(self.tmp, "home.png")
        Image.new("RGB", (1440, 900), (255, 255, 255)).save(shot_img)
        shots = [
            {"id": "s01", "type": "TALK", "narration": "مرحبا", "location": "studio", "action": "", "camera": ""},
            {"id": "s02", "type": "POINT", "narration": "هذه الصفحة", "location": "Home", "place": "Home"},
            {"id": "s03", "type": "BROLL", "narration": "والنتائج", "location": "Home", "place": "Home"},
        ]
        places = {"Home": {"path": shot_img, "screen": True}}
        with mock.patch.object(job_package.voice, "create_subtitle", _fake_subtitle):
            folder = job_package.build_package("job1", presenter, shots, {"aspect": "16:9"}, places, tts=_fake_tts)
        return folder, shots

    def test_package_contents(self):
        folder, shots = self._build()
        with open(os.path.join(folder, "job.json"), encoding="utf-8") as fp:
            job = json.load(fp)
        self.assertEqual(job["presenter"]["reference_images"], ["presenter/ref1.png"])
        self.assertAlmostEqual(job["shots"][0]["duration"], 2.0, delta=0.1)
        self.assertEqual(job["shots"][1]["location_image"], "locations/place01.png")
        self.assertEqual(job["shots"][2]["location_image"], "locations/place01.png")
        self.assertTrue(job["shots"][1]["screen"])
        self.assertNotIn("local", job["shots"][1])
        self.assertTrue(job["shots"][2]["local"])
        for shot in job["shots"]:
            self.assertTrue(os.path.isfile(os.path.join(folder, shot["audio"])))
        # The cloud worker accepts the package as it is.
        import importlib.util

        spec = importlib.util.spec_from_file_location("pw", os.path.join(utils.root_dir(), "kaggle",
                                                                         "presenter_worker.py"))
        pw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pw)
        loaded = pw.load_job(folder)
        self.assertEqual([s["id"] for s in pw.cloud_shots(loaded)], ["s01", "s02"])
        self.assertIn("wall screen", pw.placement_prompt(loaded["shots"][1], loaded["presenter"], True))

    def test_assemble_mixes_ai_screen_and_fallback_shots(self):
        folder, shots = self._build()
        root = job_package.job_dir("job1")
        os.makedirs(os.path.join(root, "output", "shots"))
        os.makedirs(os.path.join(root, "output", "frames"))
        subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=s=832x480:r=25:d=2.2", "-pix_fmt", "yuv420p",
                        os.path.join(root, "output", "shots", "s01.mp4")], check=True)
        Image.new("RGB", (832, 480), (20, 120, 200)).save(os.path.join(root, "output", "frames", "s02.png"))
        params = VideoParams(video_subject="x", subtitle_enabled=True, bgm_type="", n_threads=2,
                             font_name="Tajawal-Bold.ttf")
        with mock.patch.dict(config.app, {"video_resolution": "720p"}):
            final = assemble.assemble("job1", params)
        with VideoFileClip(final) as clip:
            self.assertEqual(clip.size, [1280, 720])
            total = sum(s["duration"] for s in shots)
            self.assertAlmostEqual(clip.duration, total, delta=0.2)
            blue = clip.get_frame(3.0)  # the fallback still of s02
            self.assertGreater(int(blue[360, 640][2]), 150)
            page = clip.get_frame(5.0)  # the website screenshot of s03
            self.assertGreater(int(page[360, 640].sum()), 600)
        with AudioFileClip(os.path.join(root, "assemble", "narration.wav")) as audio:
            self.assertAlmostEqual(audio.duration, total, delta=0.05)
        subs = assemble.read_srt(os.path.join(root, "assemble", "subtitles.srt"))
        self.assertEqual(len(subs), 3)
        self.assertAlmostEqual(subs[1][0], shots[0]["duration"], delta=0.01)


class FakeKaggleApi:
    def __init__(self, runs):
        self.runs = runs  # list of shot-id lists finished in each run
        self.calls = []
        self.run = -1
        self.pushed = []

    def get_config_value(self, name):
        return "maya" if name == "username" else None

    def dataset_create_new(self, folder, public, quiet, dir_mode):
        self.calls.append(("create", public))
        self._check_upload(folder)

    def dataset_create_version(self, folder, notes, quiet, dir_mode):
        self.calls.append(("version", notes))
        self._check_upload(folder)

    def _check_upload(self, folder):
        with zipfile.ZipFile(os.path.join(folder, "package.zip")) as zf:
            self.last_zip = sorted(zf.namelist())

    def dataset_status(self, ref):
        return "ready"

    def kernels_push(self, folder):
        with open(os.path.join(folder, "kernel-metadata.json")) as fp:
            self.pushed.append(json.load(fp))
        self.run += 1
        self.polls = 0

    def kernels_status(self, kernel):
        self.polls += 1
        return SimpleNamespace(status=SimpleNamespace(name="RUNNING" if self.polls < 2 else "COMPLETE"))

    def kernels_output(self, kernel, path, force, quiet, page_token):
        done = self.runs[self.run]
        os.makedirs(os.path.join(path, "shots"), exist_ok=True)
        for shot in done:
            open(os.path.join(path, "shots", f"{shot}.mp4"), "wb").write(b"x")
        with open(os.path.join(path, "progress.json"), "w") as fp:
            json.dump({"job_id": "job1"}, fp)
        all_done = sorted(os.listdir(os.path.join(path, "shots")))
        with open(os.path.join(path, "summary.json"), "w") as fp:
            json.dump({"total": 3, "done": len(all_done), "complete": len(all_done) == 3}, fp)
        return [], None


class TestKaggleAgent(StorageCase):
    def test_run_until_complete_with_resume(self):
        presenter = self._presenter()
        shots = [{"id": f"s0{i}", "type": "TALK", "narration": "مرحبا"} for i in (1, 2, 3)]
        with mock.patch.object(job_package.voice, "create_subtitle", _fake_subtitle):
            job_package.build_package("job1", presenter, shots, {"aspect": "9:16"}, tts=_fake_tts)
        api = FakeKaggleApi([["s01", "s02"], ["s03"]])
        messages = []
        agent = kaggle_agent.KaggleAgent(token="t", api=api, poll_seconds=0, log=messages.append,
                                         sleep=lambda s: None)
        summary = agent.run_job("job1")
        self.assertTrue(summary["complete"])
        self.assertEqual(api.calls, [("create", False), ("version", "resume")])
        meta = api.pushed[0]
        self.assertEqual(meta["id"], "maya/mpt-presenter-job1")
        self.assertEqual(meta["dataset_sources"], ["maya/mpt-job-job1"])
        self.assertTrue(meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"])
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        # The second upload carries the finished shots of the first run.
        self.assertIn("previous/shots/s01.mp4", api.last_zip)
        self.assertIn("job.json", api.last_zip)
        root = job_package.job_dir("job1")
        self.assertEqual(sorted(os.listdir(os.path.join(root, "output", "shots"))),
                         ["s01.mp4", "s02.mp4", "s03.mp4"])
        self.assertTrue(any("2/3 shots ready" in m for m in messages))

    def test_status_names(self):
        from enum import Enum

        class KernelWorkerStatus(Enum):
            COMPLETE = 2

        self.assertEqual(kaggle_agent.status_name(SimpleNamespace(status=KernelWorkerStatus.COMPLETE)), "complete")
        self.assertEqual(kaggle_agent.status_name("KernelWorkerStatus.RUNNING"), "running")

    def test_missing_token(self):
        with self.assertRaises(kaggle_agent.KaggleError):
            kaggle_agent.KaggleAgent(token="").api


if __name__ == "__main__":
    unittest.main()
