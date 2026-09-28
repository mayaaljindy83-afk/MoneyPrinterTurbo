import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from app.services.presenter import motion_library as ml
from app.utils import utils

FFMPEG = utils.get_ffmpeg_binary()


def make_clip(path, seconds=2.0, size="640x360", box_left=True, rate=25):
    """A clip with a white box on one side: stands in for an arm pointing to that side."""
    x = "20" if box_left else "iw-220"
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=black:size={size}:rate={rate}:duration={seconds}", "-vf",
                    f"drawbox=x={x}:y=40:w=200:h=200:color=white:t=fill", "-pix_fmt", "yuv420p", path], check=True)


def frames_of(path):
    probe = subprocess.run([FFMPEG, "-i", path, "-map", "0:v", "-f", "null", "-"], capture_output=True, text=True)
    raw = subprocess.run([FFMPEG, "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    w, h = ml.SIZE
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w), probe.stderr


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()

    def tearDown(self):
        self.env.stop()


class TestCatalog(Case):
    def test_all_requested_motions_with_metadata(self):
        expected = {"IDLE_PRESENT", "TALK_CAMERA", "POINT_LEFT", "POINT_RIGHT", "POINT_UP", "POINT_DOWN",
                    "PRESENT_LEFT_CARD", "PRESENT_RIGHT_CARD", "LOOK_LEFT", "LOOK_RIGHT", "TURN_LEFT", "TURN_RIGHT",
                    "WALK_LEFT", "WALK_RIGHT", "WALK_FORWARD", "WALK_AND_STOP", "WELCOME", "CTA_GESTURE"}
        self.assertEqual(set(ml.CATALOG), expected)
        meta = ml.metadata("POINT_LEFT", duration=2.5)
        self.assertEqual(meta["motion"], "POINT_LEFT")
        self.assertEqual((meta["target_side"], meta["safe_presenter_position"]), ("left", "right"))
        self.assertEqual(meta["duration"], 2.5)
        self.assertTrue(meta["camera_safe"])
        with self.assertRaises(ml.MotionError):
            ml.metadata("DANCE")

    def test_target_is_always_on_the_other_side_of_the_presenter(self):
        for motion, meta in ml.CATALOG.items():
            if meta["target_side"] in ("left", "right"):
                self.assertEqual(meta["safe_presenter_position"], ml.MIRROR[meta["target_side"]], motion)

    def test_mirror_names(self):
        self.assertEqual(ml.mirror_name("POINT_LEFT"), "POINT_RIGHT")
        self.assertEqual(ml.mirror_name("PRESENT_RIGHT_CARD"), "PRESENT_LEFT_CARD")
        self.assertEqual(ml.mirror_name("WELCOME"), "")
        self.assertEqual(ml.mirror_name("POINT_UP"), "")


class TestClips(Case):
    def test_import_normalises_and_mirror_points_the_other_way(self):
        source = os.path.join(self.tmp, "phone.mp4")
        make_clip(source, seconds=2.0, box_left=False)  # recorded pointing to the RIGHT of the frame
        path = ml.import_clip("POINT_RIGHT", source, source_note="phone recording", tested=True)
        frames, info = frames_of(path)
        self.assertEqual(frames.shape[1:], (ml.SIZE[1], ml.SIZE[0]))  # portrait 480x832
        self.assertIn("30 fps", info)
        self.assertEqual(ml.load_meta("POINT_RIGHT")["source"], "phone recording")

        status = ml.available()
        self.assertEqual(status["POINT_RIGHT"], {"source": "clip", "tested": True})
        self.assertEqual(status["POINT_LEFT"]["source"], "mirror of POINT_RIGHT")
        self.assertEqual(status["WELCOME"]["source"], "")

        right = os.path.join(self.tmp, "right.mp4")
        left = os.path.join(self.tmp, "left.mp4")
        self.assertFalse(ml.prepare("POINT_RIGHT", 2.0, right)["mirrored"])
        info = ml.prepare("POINT_LEFT", 2.0, left)
        self.assertTrue(info["mirrored"])
        self.assertEqual((info["target_side"], info["tested"]), ("left", True))
        r, _ = frames_of(right)
        lft, _ = frames_of(left)
        half = ml.SIZE[0] // 2
        # The "arm" (bright box) is on the right half in the original, on the left half when mirrored.
        self.assertGreater(r[0][:, half:].mean(), r[0][:, :half].mean())
        self.assertGreater(lft[0][:, :half].mean(), lft[0][:, half:].mean())

    def test_prepare_holds_the_last_frame_and_cuts_exactly(self):
        source = os.path.join(self.tmp, "short.mp4")
        make_clip(source, seconds=1.0)
        ml.import_clip("WELCOME", source)
        longer = os.path.join(self.tmp, "long.mp4")
        info = ml.prepare("WELCOME", 2.5, longer)
        frames, _ = frames_of(longer)
        self.assertEqual(info["frames"], 75)
        self.assertEqual(len(frames), 75)  # 1 s of clip, then the last pose is held
        shorter = os.path.join(self.tmp, "cut.mp4")
        ml.prepare("WELCOME", 0.5, shorter)
        self.assertEqual(len(frames_of(shorter)[0]), 15)

    def test_missing_clip_explains_what_to_add(self):
        with self.assertRaises(ml.MotionError) as ctx:
            ml.prepare("CTA_GESTURE", 3, os.path.join(self.tmp, "x.mp4"))
        self.assertIn("Motion Library", str(ctx.exception))

    def test_library_lives_in_the_data_folder(self):
        self.assertTrue(ml.root().startswith(self.tmp))
        ml.mark_tested("POINT_UP", notes="good hand")
        with open(os.path.join(ml.root(), "POINT_UP", "meta.json"), encoding="utf-8") as fp:
            self.assertEqual(json.load(fp)["notes"], "good hand")


if __name__ == "__main__":
    unittest.main()


class TestPackageCarriesDrivingClip(Case):
    def test_animate_shot_packaged_with_mirrored_clip(self):
        from PIL import Image

        from app.config import config
        from app.services import voice
        from app.services.presenter import package as job_package
        from app.services.presenter import profiles
        from test.services.test_presenter_local import _fake_subtitle, _fake_tts

        source = os.path.join(self.tmp, "point.mp4")
        make_clip(source, seconds=2.0, box_left=False)
        ml.import_clip("POINT_RIGHT", source, tested=True)
        photo = os.path.join(self.tmp, "lina.png")
        Image.new("RGB", (600, 900), (190, 160, 150)).save(photo)
        with mock.patch.dict(config.app, {"presenters_dir": ""}), \
                mock.patch.object(voice, "create_subtitle", _fake_subtitle):
            presenter = profiles.save_presenter(profiles.Presenter(name="Lina"), [photo])
            shots = [{"id": "s02", "type": "ANIMATE", "motion": "POINT_LEFT", "narration": "انظروا هنا",
                      "min_duration": 3.0}]
            folder = job_package.build_package("job1", presenter, shots, {"aspect": "16:9"}, tts=_fake_tts)
        with open(os.path.join(folder, "job.json"), encoding="utf-8") as fp:
            entry = json.load(fp)["shots"][0]
        self.assertEqual(entry["driving"], "motions/s02.mp4")
        self.assertTrue(entry["mirrored"])  # POINT_LEFT made from the POINT_RIGHT recording
        self.assertTrue(entry["green"])  # keyed and composited on the laptop
        self.assertIn("left side", entry["pose_prompt"])
        frames, _ = frames_of(os.path.join(folder, entry["driving"]))
        self.assertEqual(len(frames), round(entry["duration"] * ml.FPS))


class TestAICandidates(Case):
    def _presenter(self):
        from PIL import Image

        from app.config import config
        from app.services.presenter import profiles

        photo = os.path.join(self.tmp, "lina.png")
        Image.new("RGB", (600, 900), (190, 160, 150)).save(photo)
        with mock.patch.dict(config.app, {"presenters_dir": ""}):
            return profiles.save_presenter(profiles.Presenter(name="Lina"), [photo])

    def test_job_has_drive_shots_one_side_only_and_distinct_seeds(self):
        from app.services.presenter import package as job_package

        plan = ml.build_candidates_job("motions1", self._presenter(), ["POINT_LEFT", "POINT_RIGHT", "WELCOME"], 2)
        self.assertEqual(sorted(set(plan["motions"].values())), ["POINT_LEFT", "WELCOME"])  # twin skipped
        with open(os.path.join(job_package.job_dir("motions1"), "package", "job.json"), encoding="utf-8") as fp:
            job = json.load(fp)
        self.assertEqual(job["settings"]["aspect"], "9:16")
        self.assertEqual({s["type"] for s in job["shots"]}, {"DRIVE"})
        self.assertEqual(len({s["seed"] for s in job["shots"]}), 4)
        self.assertIn("left side", job["shots"][0]["action"])
        self.assertEqual(job_package.load_plan("motions1")["kind"], "motion_candidates")

    def test_studio_run_keeps_candidates_and_skips_assembly(self):
        from app.config import config
        from app.services.presenter import package as job_package
        from app.services.presenter import studio

        self._presenter()

        class FakeAgent:
            def run_job(self, job_id, on_status=None, continuing=False):
                with open(os.path.join(job_package.job_dir(job_id), "package", "job.json"), encoding="utf-8") as fp:
                    shots = json.load(fp)["shots"]
                out = os.path.join(job_package.job_dir(job_id), "output", "shots")
                os.makedirs(out, exist_ok=True)
                for shot in shots:
                    make_clip(os.path.join(out, f"{shot['id']}.mp4"), seconds=1.0, box_left=False)
                return {"done": len(shots), "total": len(shots), "complete": True}

        with mock.patch.dict(config.app, {"presenters_dir": ""}), \
                mock.patch.object(studio, "render_final", side_effect=AssertionError("no assembly")):
            job_id = studio.job_package.new_job_id("motions")
            ml.build_candidates_job(job_id, __import__("app.services.presenter.profiles", fromlist=["x"])
                                    .load_presenter("Lina"), ["POINT_RIGHT"], 2)
            studio._run_on_kaggle(job_id, "", {}, agent=FakeAgent())
        self.assertEqual(studio.read_status(job_id)["state"], "done")
        found = ml.candidates(job_id)
        self.assertEqual([c["motion"] for c in found], ["POINT_RIGHT", "POINT_RIGHT"])
        ml.use_candidate("POINT_RIGHT", found[1]["path"])
        self.assertEqual(ml.available()["POINT_LEFT"]["source"], "mirror of POINT_RIGHT")
        self.assertFalse(ml.load_meta("POINT_RIGHT")["tested"])  # tested only after a good render

    def test_search_links(self):
        self.assertTrue(ml.search_link("POINT_LEFT").startswith("https://www.pexels.com/search/videos/"))
        self.assertIn("portrait", ml.search_link("WELCOME"))


class TestMotionLibraryPage(Case):
    def test_page_lists_all_motions_and_needs_a_gpu_service(self):
        from streamlit.testing.v1 import AppTest

        from app.config import config

        page = os.path.join(os.path.dirname(__file__), "..", "..", "webui", "pages", "3_Motion_Library.py")
        with mock.patch.dict(config.app, {"presenter_cloud": "kaggle", "kaggle_api_token": ""}):
            app = AppTest.from_file(page, default_timeout=60)
            app.run()
        self.assertFalse(app.exception, app.exception)
        self.assertEqual(len(app.dataframe[0].value), 18)
        make = next(b for b in app.button if b.label == "اعملي الفيديوهات")
        self.assertTrue(make.disabled)
        self.assertIn("جهّزي Kaggle أو RunPod بصفحة Presenter Video أول.", [i.value for i in app.info])
