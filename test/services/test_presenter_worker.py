import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from moviepy import VideoFileClip
from PIL import Image

from app.utils import utils

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FFMPEG = utils.get_ffmpeg_binary()

_spec = importlib.util.spec_from_file_location("presenter_worker", os.path.join(ROOT, "kaggle", "presenter_worker.py"))
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)


def _links_are_valid(test, graph):
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                test.assertIn(value[0], graph)


class TestFramePlanning(unittest.TestCase):
    def test_talk_segments_cover_audio_with_overlap(self):
        self.assertEqual(pw.talk_segments(2), (1, 81))
        self.assertEqual(pw.talk_segments(3.2), (1, 81))
        for seconds in (3.3, 6, 10, 31.7, 90):
            segments, frames = pw.talk_segments(seconds)
            self.assertGreaterEqual(frames, seconds * pw.TALK_FPS)
            # One segment fewer would not be enough.
            self.assertLess(pw.SEGMENT_FRAMES + (segments - 2) * 72, seconds * pw.TALK_FPS + 1)
            self.assertEqual(frames, 81 + (segments - 1) * 72)

    def test_motion_chunks_are_4k_plus_1_and_cover_duration(self):
        for seconds in (0.5, 3, 5, 5.1, 12, 30):
            chunks = pw.motion_chunks(seconds)
            for frames in chunks:
                self.assertEqual((frames - 1) % 4, 0)
                self.assertLessEqual(frames, 81)
                self.assertGreaterEqual(frames, 17)
            total = chunks[0] + sum(c - 1 for c in chunks[1:])
            self.assertGreaterEqual(total, seconds * pw.MOTION_FPS)

    def test_video_size(self):
        self.assertEqual(pw.video_size({"settings": {"aspect": "9:16"}}), (480, 832))
        self.assertEqual(pw.video_size({}), (832, 480))


class TestJobValidation(unittest.TestCase):
    def _job(self, **changes):
        job = {"job_id": "j", "presenter": {"reference_images": ["presenter/ref1.png"]},
               "shots": [{"id": "s01", "type": "TALK", "duration": 4, "audio": "audio/s01.mp3"}]}
        job.update(changes)
        return job

    def test_valid(self):
        pw.validate_job(self._job())

    def test_errors(self):
        bad = [
            self._job(shots=[]),
            self._job(shots=[{"id": "s 1", "type": "TALK", "duration": 1, "audio": "a"}]),
            self._job(shots=[{"id": "a", "type": "WALK", "duration": 1}, {"id": "a", "type": "WALK", "duration": 1}]),
            self._job(shots=[{"id": "a", "type": "DANCE", "duration": 1}]),
            self._job(shots=[{"id": "a", "type": "TALK", "duration": 1}]),
            self._job(shots=[{"id": "a", "type": "WALK", "duration": 0}]),
            self._job(presenter={}),
        ]
        for job in bad:
            with self.assertRaises(SystemExit):
                pw.validate_job(job)

    def test_broll_only_job_needs_no_presenter(self):
        pw.validate_job(self._job(presenter={}, shots=[{"id": "b", "type": "BROLL", "duration": 3}]))

    def test_create_presenter_job(self):
        pw.validate_job({"kind": "create_presenter", "prompt": "a woman"})
        with self.assertRaises(SystemExit):
            pw.validate_job({"kind": "create_presenter"})

    def test_load_job_finds_nested_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "pkg"))
            with open(os.path.join(tmp, "pkg", "job.json"), "w") as fp:
                json.dump(self._job(), fp)
            job = pw.load_job(tmp)
            self.assertEqual(job["_root"], os.path.join(tmp, "pkg"))

    def test_load_job_from_uploaded_zip(self):
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "input", "datasets", "maya", "mpt-job-x"))
            with zipfile.ZipFile(os.path.join(tmp, "input", "datasets", "maya", "mpt-job-x", "package.zip"),
                                 "w") as zf:
                zf.writestr("job.json", json.dumps(self._job()))
                zf.writestr("presenter/ref1.png", b"png")
            job = pw.load_job(os.path.join(tmp, "input"), os.path.join(tmp, "work"))
            self.assertEqual(job["_root"], os.path.join(tmp, "work", "job"))
            self.assertTrue(os.path.isfile(os.path.join(job["_root"], "presenter", "ref1.png")))

    def test_local_shots_are_not_rendered_in_the_cloud(self):
        job = self._job(shots=[{"id": "a", "type": "TALK", "duration": 2, "audio": "x"},
                               {"id": "b", "type": "BROLL", "duration": 2, "local": True}])
        self.assertEqual([s["id"] for s in pw.cloud_shots(job)], ["a"])


class TestWorkflows(unittest.TestCase):
    def test_talk_chain_trims_overlap_and_batches(self):
        graph = pw.build_talk_workflow("s.png", "a.wav", "p", 832, 480, 3, "talk", 5)
        _links_are_valid(self, graph)
        self.assertNotIn("previous_frames", graph["s0_talk"]["inputs"])
        self.assertEqual(graph["s1_talk"]["inputs"]["previous_frames"], ["s0_decode", 0])
        self.assertEqual(graph["s2_talk"]["inputs"]["previous_frames"], ["s1_all", 0])
        self.assertEqual(graph["s1_trim"]["inputs"]["batch_index"], pw.MOTION_CONTEXT)
        self.assertEqual(graph["s2_all"]["inputs"]["images.image0"], ["s1_all", 0])
        self.assertEqual(graph["save"]["inputs"]["images"], ["s2_all", 0])
        self.assertEqual(graph["s0_talk"]["inputs"]["mode"], "single_speaker")
        self.assertEqual(graph["s1_noise"]["inputs"]["noise_seed"], 6)

    def test_single_segment_saves_decode(self):
        graph = pw.build_talk_workflow("s.png", "a.wav", "p", 480, 832, 1, "talk", 5)
        self.assertEqual(graph["save"]["inputs"]["images"], ["s0_decode", 0])
        self.assertEqual(graph["s0_talk"]["inputs"]["width"], 480)

    def test_edit_workflow_reference_images(self):
        one = pw.build_edit_workflow("p", "a.png", None, None, "f", 1)
        self.assertNotIn("image2", one["pos_text"]["inputs"])
        three = pw.build_edit_workflow("p", "a.png", "b.png", "c.png", "f", 1)
        _links_are_valid(self, three)
        self.assertEqual(three["neg_text"]["inputs"]["image3"], ["img3", 0])
        self.assertEqual(three["sampler"]["inputs"]["steps"], 4)

    def test_motion_and_portrait_workflows(self):
        motion = pw.build_motion_workflow("s.png", "walks", 832, 480, 49, "m", 1)
        _links_are_valid(self, motion)
        self.assertEqual(motion["i2v"]["inputs"]["length"], 49)
        portrait = pw.build_portrait_workflow("a woman", 832, 1216, "c", 1)
        _links_are_valid(self, portrait)

    def test_workflow_models_are_known(self):
        graphs = [pw.build_talk_workflow("s", "a", "p", 832, 480, 2, "t", 1),
                  pw.build_edit_workflow("p", "a", "b", None, "f", 1),
                  pw.build_motion_workflow("s", "p", 832, 480, 81, "m", 1),
                  pw.build_portrait_workflow("p", 832, 1216, "c", 1)]
        for graph in graphs:
            for node in graph.values():
                for key in ("unet_name", "lora_name", "clip_name", "vae_name", "name", "audio_encoder_name"):
                    if key in node["inputs"]:
                        self.assertIn(node["inputs"][key], pw.MODELS)

    def test_prompts(self):
        presenter = {"description": "a 25-year-old woman in a navy blazer"}
        shot = {"action": "points at the screen", "camera": "medium shot", "location": "a modern office"}
        self.assertIn("image 2", pw.placement_prompt(shot, presenter, True))
        self.assertIn("a modern office", pw.placement_prompt(shot, presenter, False))
        self.assertIn("points at the screen", pw.talk_prompt(shot))
        self.assertIn("no people", pw.broll_prompt(shot))
        self.assertEqual(pw.motion_prompt({}), "Natural smooth motion.")


# Input names of the ComfyUI v0.37.4 core nodes the Wan Animate 2 graph uses (from the node sources).
NODE_INPUTS = {
    "WanAnimate2ToVideo": {"positive", "negative", "vae", "width", "height", "length", "batch_size",
                           "reference_image", "pose_video", "clip_vision_output", "positive_pose",
                           "clip_vision_output_pose", "continue_motion", "video_frame_offset", "pose_strength",
                           "pose_start_percent", "pose_end_percent", "reference_image_strength"},
    "WanAnimate2Cache": {"model", "device", "dtype"},
    "LoadVideo": {"file"},
    "GetVideoComponents": {"video"},
    "TrimVideoLatent": {"samples", "trim_amount"},
    "ImageFromBatch": {"image", "batch_index", "length"},
    "SamplerCustom": {"model", "add_noise", "noise_seed", "cfg", "positive", "negative", "sampler", "sigmas",
                      "latent_image"},
    "KSamplerSelect": {"sampler_name"},
    "BasicScheduler": {"model", "scheduler", "steps", "denoise"},
    "ModelSamplingSD3": {"model", "shift"},
    "CLIPVisionEncode": {"clip_vision", "image", "crop"},
}


class TestAnimateWorkflow(unittest.TestCase):
    def test_segments_cover_frames_with_legal_lengths(self):
        self.assertEqual(pw.animate_segments(81), [81])
        self.assertEqual(pw.animate_segments(60), [61])
        self.assertEqual(pw.animate_segments(300), [81, 81, 81, 61])
        for frames in (1, 30, 81, 82, 150, 161, 300, 450):
            lengths = pw.animate_segments(frames)
            self.assertTrue(all(n % 4 == 1 and n <= 81 for n in lengths), lengths)
            made = lengths[0] + sum(n - 1 for n in lengths[1:])  # joins repeat one frame
            self.assertGreaterEqual(made, frames)
            self.assertLess(made - frames, 4 + 1)

    def test_graph_matches_official_template_and_node_inputs(self):
        g = pw.build_animate_workflow("ref.png", "drive.mp4", "Character Description: x", "a woman points left",
                                      480, 832, 300, "animate_s01", 7)
        _links_are_valid(self, g)
        for node in g.values():
            known = NODE_INPUTS.get(node["class_type"])
            if known:
                self.assertLessEqual(set(node["inputs"]), known, node["class_type"])
        # Template settings: lightx2v, 6 steps, lcm, shift 5, cfg 1.
        self.assertEqual(g["sigmas"]["inputs"]["steps"], 6)
        self.assertEqual(g["sampler_select"]["inputs"]["sampler_name"], "lcm")
        self.assertEqual(g["shift"]["inputs"]["shift"], 5.0)
        self.assertEqual(g["a0_sample"]["inputs"]["cfg"], 1.0)
        # Chunks chain: continue from the frames so far, pose video continues where the last chunk ended.
        self.assertNotIn("continue_motion", g["a0_anim"]["inputs"])
        self.assertEqual(g["a1_anim"]["inputs"]["continue_motion"], ["a0_decode", 0])
        self.assertEqual(g["a1_anim"]["inputs"]["video_frame_offset"], ["a0_anim", 5])
        self.assertEqual(g["a2_anim"]["inputs"]["video_frame_offset"], ["a1_anim", 5])
        self.assertEqual(g["a1_new"]["inputs"]["batch_index"], ["a1_anim", 4])  # joining frame dropped
        self.assertEqual(g["a3_anim"]["inputs"]["length"], 61)
        self.assertEqual(g["exact"]["inputs"]["length"], 300)
        self.assertEqual(g["save"]["inputs"]["images"], ["exact", 0])
        for node in g.values():
            for key in ("unet_name", "lora_name", "clip_name", "vae_name"):
                if key in node["inputs"]:
                    self.assertIn(node["inputs"][key], pw.MODELS)
        self.assertEqual(set(pw.ANIMATE_MODELS) - set(pw.MODELS), set())

    def test_prompt_has_looks_and_green_background_but_no_motion(self):
        prompt = pw.animate_prompt({"description": "a woman in a navy blazer"})
        self.assertIn("navy blazer", prompt)
        self.assertIn("#00FF00", prompt)
        self.assertNotIn("point", prompt.lower())

    def test_animate_shot_needs_a_driving_clip(self):
        job = {"job_id": "j", "presenter": {"reference_images": ["r.png"]},
               "shots": [{"id": "s01", "type": "ANIMATE", "duration": 3, "audio": "a.mp3"}]}
        with self.assertRaises(SystemExit):
            pw.validate_job(job)
        job["shots"][0]["driving"] = "motions/s01.mp4"
        pw.validate_job(job)


class TestMediaHelpers(unittest.TestCase):
    def test_fit_image_crop_and_pad(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "p.png")
            Image.new("RGB", (600, 900), (255, 0, 0)).save(src)
            cropped = Image.open(pw.fit_image(src, os.path.join(tmp, "c.png"), 832, 480, pad_color=None))
            self.assertAlmostEqual(cropped.width / cropped.height, 832 / 480, delta=0.02)
            self.assertEqual(cropped.getpixel((5, 5)), (255, 0, 0))
            padded = Image.open(pw.fit_image(src, os.path.join(tmp, "d.png"), 832, 480))
            self.assertAlmostEqual(padded.width / padded.height, 832 / 480, delta=0.02)
            self.assertEqual(padded.getpixel((5, 5)), (128, 128, 128))
            self.assertEqual(padded.getpixel((padded.width // 2, padded.height // 2)), (255, 0, 0))


class TestProgressAndResume(unittest.TestCase):
    def test_progress_rate_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = pw.Progress(tmp, {"job_id": "j"})
            self.assertIsNone(progress.seconds_per_video_second())
            progress.set("s1", status="done", render_seconds=600, duration=10)
            progress.set("s2", status="failed", render_seconds=5, duration=10)
            self.assertEqual(pw.Progress(tmp, {"job_id": "j"}).seconds_per_video_second(), 60)
            self.assertEqual(pw.Progress(tmp, {"job_id": "other"}).data["shots"], {})

    def test_restore_previous_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.path.join(tmp, "input", "prev-run")
            os.makedirs(os.path.join(previous, "shots"))
            with open(os.path.join(previous, "progress.json"), "w") as fp:
                json.dump({"job_id": "j", "shots": {"s1": {"status": "done"}}}, fp)
            open(os.path.join(previous, "shots", "s1.mp4"), "wb").close()
            other = os.path.join(tmp, "input", "other")
            os.makedirs(os.path.join(other, "shots"))
            with open(os.path.join(other, "progress.json"), "w") as fp:
                json.dump({"job_id": "x"}, fp)
            open(os.path.join(other, "shots", "s9.mp4"), "wb").close()
            out = os.path.join(tmp, "out")
            self.assertEqual(pw.restore_previous_outputs(out, [os.path.join(tmp, "input")], "j"), 1)
            self.assertEqual(os.listdir(os.path.join(out, "shots")), ["s1.mp4"])
            self.assertIn("s1", pw.Progress(out, {"job_id": "j"}).data["shots"])


class FakeComfy:
    """Stands in for ComfyUI: writes the frames a graph would produce."""

    def __init__(self, comfy_dir):
        self.dir = comfy_dir
        self.graphs = []
        self.fail = set()

    input_dir = property(lambda self: os.path.join(self.dir, "input"))
    output_dir = property(lambda self: os.path.join(self.dir, "output"))

    def start(self, extra_args=None):
        pass

    def stop(self):
        pass

    def run(self, graph, timeout=0):
        self.graphs.append(graph)
        prefix = graph["save"]["inputs"]["filename_prefix"]
        if prefix in self.fail:
            raise RuntimeError("CUDA out of memory")
        for node in graph.values():
            if node["class_type"] == "LoadImage":
                assert os.path.isfile(os.path.join(self.input_dir, node["inputs"]["image"]))
        if "s0_talk" in graph:
            talk = graph["s0_talk"]["inputs"]
            segments = sum(1 for k in graph if k.endswith("_talk"))
            count, size = 81 + (segments - 1) * 72, (talk["width"], talk["height"])
        elif "exact" in graph:  # Wan Animate 2
            anim = graph["a0_anim"]["inputs"]
            count, size = graph["exact"]["inputs"]["length"], (anim["width"], anim["height"])
            assert os.path.isfile(os.path.join(self.input_dir, graph["drive_file"]["inputs"]["file"]))
        elif "i2v" in graph:
            i2v = graph["i2v"]["inputs"]
            count, size = i2v["length"], (i2v["width"], i2v["height"])
        else:
            count, size = 1, (1024, 592)
        os.makedirs(self.output_dir, exist_ok=True)
        paths = []
        for i in range(count):
            path = os.path.join(self.output_dir, f"{prefix}_{i + 1:05d}_.png")
            Image.new("RGB", size, (i * 3 % 255, 90, 160)).save(path)
            paths.append(path)
        return paths


class TestWorkerOrchestration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.job_dir = os.path.join(self.tmp, "job")
        for sub in ("presenter", "audio", "locations"):
            os.makedirs(os.path.join(self.job_dir, sub))
        Image.new("RGB", (512, 768), (200, 180, 170)).save(os.path.join(self.job_dir, "presenter", "ref1.png"))
        Image.new("RGB", (1600, 900), (30, 90, 30)).save(os.path.join(self.job_dir, "locations", "park.jpg"))
        subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=4",
                        os.path.join(self.job_dir, "audio", "s01.mp3")], check=True)
        self.job = {
            "job_id": "demo", "settings": {"aspect": "16:9", "seed": 7},
            "presenter": {"reference_images": ["presenter/ref1.png"], "description": "a young woman"},
            "shots": [
                {"id": "s01", "type": "TALK", "duration": 4, "audio": "audio/s01.mp3", "location": "a studio"},
                {"id": "s02", "type": "WALK", "duration": 6, "location_image": "locations/park.jpg"},
                {"id": "s03", "type": "BROLL", "duration": 2, "location_image": "locations/park.jpg"},
            ],
        }
        with open(os.path.join(self.job_dir, "job.json"), "w") as fp:
            json.dump(self.job, fp)

    def _worker(self, fake):
        job = pw.load_job(self.job_dir)
        worker = pw.Worker(job, os.path.join(self.tmp, "out"), os.path.join(self.tmp, "work"),
                           os.path.join(self.tmp, "ComfyUI"), [], time_budget=10 * 3600)
        worker.comfy = fake
        return worker

    def test_full_run_then_resume(self):
        fake = FakeComfy(os.path.join(self.tmp, "ComfyUI"))
        fake.fail.add("motion_s02_1")
        with mock.patch.object(pw, "ensure_models"), mock.patch.object(pw, "free_models"):
            worker = self._worker(fake)
            worker.make_frames()
            # The BROLL shot with a photo uses the photo; the others go through Qwen edit.
            edit_prefixes = [g["save"]["inputs"]["filename_prefix"] for g in fake.graphs]
            self.assertEqual(edit_prefixes, ["frame_s01", "frame_s02"])
            self.assertIn("image2", fake.graphs[1]["pos_text"]["inputs"])  # location + presenter
            worker.make_videos()
            summary = pw.summarize(worker)
        self.assertEqual(summary["remaining"], ["s02"])
        out = os.path.join(self.tmp, "out")
        with VideoFileClip(os.path.join(out, "shots", "s01.mp4")) as clip:
            self.assertEqual(clip.size, [832, 480])
            self.assertAlmostEqual(clip.duration, 4, delta=0.15)
            self.assertAlmostEqual(clip.fps, 25, delta=0.1)
            self.assertIsNotNone(clip.audio)
        with VideoFileClip(os.path.join(out, "shots", "s03.mp4")) as clip:
            self.assertAlmostEqual(clip.duration, 2, delta=0.15)
        progress = json.load(open(os.path.join(out, "progress.json")))
        self.assertEqual(progress["shots"]["s02"]["status"], "failed")

        # Rerun: only the failed shot is rendered, as two chained chunks.
        fake2 = FakeComfy(os.path.join(self.tmp, "ComfyUI"))
        with mock.patch.object(pw, "ensure_models"), mock.patch.object(pw, "free_models"):
            worker = self._worker(fake2)
            worker.make_frames()
            worker.make_videos()
            self.assertTrue(pw.summarize(worker)["complete"])
        self.assertEqual([g["save"]["inputs"]["filename_prefix"] for g in fake2.graphs],
                         ["motion_s02_0", "motion_s02_1"])
        with VideoFileClip(os.path.join(out, "shots", "s02.mp4")) as clip:
            self.assertAlmostEqual(clip.duration, 6, delta=0.15)
            self.assertEqual(clip.size, [832, 480])

    def test_animate_shot_renders_portrait_30fps_without_first_frame(self):
        os.makedirs(os.path.join(self.job_dir, "motions"))
        subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=480x832:rate=30:duration=3",
                        "-pix_fmt", "yuv420p", os.path.join(self.job_dir, "motions", "s04.mp4")], check=True)
        self.job["shots"] = [{"id": "s04", "type": "ANIMATE", "duration": 3, "audio": "audio/s01.mp3",
                              "driving": "motions/s04.mp4", "pose_prompt": "a woman points left", "green": True}]
        with open(os.path.join(self.job_dir, "job.json"), "w") as fp:
            json.dump(self.job, fp)
        fake = FakeComfy(os.path.join(self.tmp, "ComfyUI"))
        with mock.patch.object(pw, "ensure_models") as models, mock.patch.object(pw, "free_models"):
            worker = self._worker(fake)
            worker.make_frames()
            self.assertEqual(fake.graphs, [])  # no Qwen first frame for ANIMATE
            worker.make_videos()
            self.assertTrue(pw.summarize(worker)["complete"])
        self.assertEqual(models.call_args[0][0], pw.ANIMATE_MODELS)  # no InfiniteTalk / I2V downloads
        graph = fake.graphs[0]
        self.assertEqual(graph["pose_text"]["inputs"]["text"], "a woman points left")
        self.assertIn("a young woman", graph["pos"]["inputs"]["text"])
        with VideoFileClip(os.path.join(self.tmp, "out", "shots", "s04.mp4")) as clip:
            self.assertEqual(clip.size, [480, 832])
            self.assertAlmostEqual(clip.fps, 30, delta=0.1)
            self.assertAlmostEqual(clip.duration, 3, delta=0.1)
            self.assertIsNotNone(clip.audio)

    def test_time_budget_stops_before_video(self):
        fake = FakeComfy(os.path.join(self.tmp, "ComfyUI"))
        with mock.patch.object(pw, "ensure_models"), mock.patch.object(pw, "free_models"):
            worker = self._worker(fake)
            worker.deadline = worker.deadline - 10 * 3600 + 300  # 5 minutes left
            worker.make_frames()
            worker.make_videos()
        self.assertEqual(fake.graphs, [])
        self.assertFalse(pw.summarize(worker)["complete"])


if __name__ == "__main__":
    unittest.main()


class TestPresenterNotebook(unittest.TestCase):
    def test_committed_colab_notebook_matches_worker(self):
        spec = importlib.util.spec_from_file_location("build_notebook", os.path.join(ROOT, "kaggle", "build_notebook.py"))
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with open(builder.PRESENTER_COLAB_NOTEBOOK, encoding="utf-8") as fp:
            committed = json.load(fp)
        self.assertEqual(committed, builder.build_presenter_colab(), "run: python kaggle/build_notebook.py")
        sources = ["".join(cell["source"]) for cell in committed["cells"]]
        self.assertTrue(any(s.startswith("%%writefile /content/presenter_worker.py") for s in sources))

    def test_colab_output_is_per_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, "job")
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, "job.json"), "w") as fp:
                json.dump({"job_id": "abc", "kind": "create_presenter", "prompt": "x"}, fp)
            paths = {"out": os.path.join(tmp, "out"), "work": os.path.join(tmp, "work"),
                     "comfy": os.path.join(tmp, "comfy"), "cache_dirs": []}
            created = []
            with mock.patch.object(pw, "detect_platform", return_value="colab"), \
                    mock.patch.object(pw, "default_paths", return_value=paths), \
                    mock.patch.object(pw.Worker, "create_presenter", lambda self: created.append(self.out)):
                pw.main(["--job", job_dir, "--skip-setup"])
            self.assertEqual(created, [os.path.join(tmp, "out", "abc")])


class TestModelDownload(unittest.TestCase):
    """Regression: Kaggle run died at 'curl: (92) HTTP/2 stream 0 was not closed cleanly' on a 20 GB model."""

    def _fake_curl(self, script):
        calls = []

        def run(command, *args, **kwargs):
            calls.append(command)
            partial = command[command.index("-o") + 1]
            chunk, code = script[len(calls) - 1]
            with open(partial, "ab") as fp:  # -C - resumes: new bytes are appended
                fp.write(b"x" * chunk)
            return subprocess.CompletedProcess(command, code)

        return run, calls

    def test_resumes_after_http2_stream_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = os.path.join(tmp, "model.part")
            fake, calls = self._fake_curl([(400, 92), (600, 0)])
            with mock.patch.object(pw.subprocess, "run", fake), mock.patch.object(pw, "remote_size", return_value=1000), \
                    mock.patch.object(pw.time, "sleep"):
                pw.download("https://huggingface.co/x/model.safetensors", partial)
            self.assertEqual(os.path.getsize(partial), 1000)
            self.assertEqual(len(calls), 2)
            for flag in ("--http1.1", "--retry-all-errors", "-C"):
                self.assertIn(flag, calls[0])

    def test_complete_file_counts_even_if_curl_complains(self):
        # Resuming an already complete file makes the server answer 416 and curl exit non-zero.
        with tempfile.TemporaryDirectory() as tmp:
            partial = os.path.join(tmp, "model.part")
            fake, calls = self._fake_curl([(1000, 92), (0, 22)])
            with mock.patch.object(pw.subprocess, "run", fake), mock.patch.object(pw, "remote_size", return_value=1000), \
                    mock.patch.object(pw.time, "sleep"):
                pw.download("https://huggingface.co/x/model.safetensors", partial)
            self.assertEqual(len(calls), 1)

    def test_models_kept_on_the_runpod_volume(self):
        """RunPod: download into the network volume once; later starts link it from there."""
        name = "clip_vision_h.safetensors"
        folder = pw.MODELS[name][1]

        def fake_download(url, partial):
            with open(partial, "wb") as fp:
                fp.write(b"w" * 2_000_000)

        with tempfile.TemporaryDirectory() as tmp:
            store, comfy = os.path.join(tmp, "volume", "models"), os.path.join(tmp, "comfy")
            with mock.patch.dict(os.environ, {"MPT_MODEL_STORE": store}), \
                    mock.patch.object(pw, "download", side_effect=fake_download) as download:
                pw.ensure_models([name], comfy, [store])
                target = os.path.join(comfy, "models", folder, name)
                self.assertTrue(os.path.islink(target))
                self.assertTrue(os.path.isfile(os.path.join(store, folder, name)))
                pw.free_models([name], comfy)  # never deletes the kept copy
                self.assertTrue(os.path.isfile(os.path.join(store, folder, name)))
                fresh = os.path.join(tmp, "comfy2")  # the next cold start
                pw.ensure_models([name], fresh, [store])
                self.assertEqual(download.call_count, 1)
                self.assertTrue(os.path.islink(os.path.join(fresh, "models", folder, name)))

    def test_gives_up_after_all_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake, calls = self._fake_curl([(10, 92)] * pw.DOWNLOAD_ATTEMPTS)
            with mock.patch.object(pw.subprocess, "run", fake), mock.patch.object(pw, "remote_size", return_value=1000), \
                    mock.patch.object(pw.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    pw.download("https://huggingface.co/x/model.safetensors", os.path.join(tmp, "m.part"))
            self.assertEqual(len(calls), pw.DOWNLOAD_ATTEMPTS)
