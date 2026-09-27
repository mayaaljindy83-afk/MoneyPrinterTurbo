import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock

from moviepy import VideoFileClip
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "kaggle", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wan_clips = _load("wan_clips")
build_notebook = _load("build_notebook")


class TestNotebook(unittest.TestCase):
    def test_committed_notebook_matches_script(self):
        with open(build_notebook.NOTEBOOK, encoding="utf-8") as fp:
            committed = json.load(fp)
        self.assertEqual(
            committed,
            build_notebook.build(),
            "run: python kaggle/build_notebook.py",
        )

    def test_committed_colab_notebook_matches_script(self):
        with open(build_notebook.COLAB_NOTEBOOK, encoding="utf-8") as fp:
            committed = json.load(fp)
        self.assertEqual(committed, build_notebook.build_colab(), "run: python kaggle/build_notebook.py")

    def test_colab_notebook_uses_fp8_and_content_paths(self):
        notebook = build_notebook.build_colab()
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        run_cell = next(
            s for s in sources if "subprocess.run(command" in s and not s.startswith("%%writefile")
        )
        self.assertIn('"--weight-dtype", "fp8_e4m3fn"', run_cell)
        self.assertIn("/content/prompts.txt", run_cell)
        self.assertTrue(any("files.download" in s for s in sources))
        self.assertFalse(any("/kaggle/" in s for s in sources if not s.startswith("%%writefile")))
        self.assertEqual(notebook["metadata"]["colab"]["gpuType"], "T4")

    def test_notebook_embeds_script(self):
        cell = "".join(build_notebook.build()["cells"][3]["source"])
        with open(build_notebook.SCRIPT, encoding="utf-8") as fp:
            self.assertEqual(cell, "%%writefile /kaggle/working/wan_clips.py\n" + fp.read().strip("\n"))


class TestSettings(unittest.TestCase):
    def test_frames_follow_wan_4k_plus_1_rule(self):
        self.assertEqual(wan_clips.frames_for(5), 121)
        for seconds in (1, 2.5, 3, 4, 7):
            self.assertEqual((wan_clips.frames_for(seconds) - 1) % 4, 0)

    def test_sizes_are_multiples_of_32(self):
        for (w, h) in wan_clips.SIZES.values():
            self.assertEqual(w % 32, 0)
            self.assertEqual(h % 32, 0)

    def test_requirements_filter_keeps_torchsde(self):
        kept = wan_clips.filter_requirements(
            ["torch", "torchvision>=0.20", "torchaudio ; python_version>'3'", "torchsde",
             "# comment", "", "numpy>=1.25.0  # inline", "Torch_Audio==1"]
        )
        self.assertEqual(kept, ["torchsde", "numpy>=1.25.0", "Torch_Audio==1"])


class TestWorkflow(unittest.TestCase):
    def test_fast_workflow(self):
        graph = wan_clips.build_workflow("a reef", "clip_001", 960, 544, 121, 7, fast=True)
        self.assertEqual(graph["2"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["3"]["inputs"]["model"], ["2", 0])
        sampler = graph["9"]["inputs"]
        self.assertEqual((sampler["steps"], sampler["cfg"]), (8, 1.0))
        self.assertEqual(graph["8"]["inputs"]["length"], 121)
        self.assertEqual(graph["5"]["inputs"]["text"], "a reef")
        # Every link points at an existing node.
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)

    def test_fp8_weights_option(self):
        graph = wan_clips.build_workflow("a reef", "p", 960, 544, 121, 7, weight_dtype="fp8_e4m3fn")
        self.assertEqual(graph["1"]["inputs"]["weight_dtype"], "fp8_e4m3fn")
        default = wan_clips.build_workflow("a reef", "p", 960, 544, 121, 7)
        self.assertEqual(default["1"]["inputs"]["weight_dtype"], "default")

    def test_official_workflow_has_no_lora(self):
        graph = wan_clips.build_workflow("a reef", "p", 1280, 704, 121, 7, fast=False)
        self.assertNotIn("2", graph)
        self.assertEqual(graph["3"]["inputs"]["model"], ["1", 0])
        self.assertEqual((graph["9"]["inputs"]["steps"], graph["9"]["inputs"]["cfg"]), (20, 5.0))


class TestGenerateLoop(unittest.TestCase):
    """Runs generate() with a fake ComfyUI that writes PNG frames."""

    def test_clips_are_encoded_skipped_and_failures_survived(self):
        with tempfile.TemporaryDirectory() as tmp:
            comfy = os.path.join(tmp, "ComfyUI")
            out = os.path.join(tmp, "clips")
            calls = []

            def fake_queue(graph, timeout=3600):
                prefix = graph["11"]["inputs"]["filename_prefix"]
                calls.append(prefix)
                if prefix == "clip_002":
                    raise RuntimeError("out of memory")
                length = graph["8"]["inputs"]["length"]
                width = graph["8"]["inputs"]["width"]
                height = graph["8"]["inputs"]["height"]
                os.makedirs(os.path.join(comfy, "output"), exist_ok=True)
                images = []
                for i in range(length):
                    name = f"{prefix}_{i + 1:05d}_.png"
                    Image.new("RGB", (width, height), (i * 2 % 255, 80, 160)).save(
                        os.path.join(comfy, "output", name)
                    )
                    images.append({"filename": name, "subfolder": "", "type": "output"})
                return {"outputs": {"11": {"images": images}}}

            with mock.patch.object(wan_clips, "COMFY_DIR", comfy), mock.patch.object(
                wan_clips, "queue_and_wait", fake_queue
            ):
                files = wan_clips.generate(
                    ["reef", "jellyfish", "submarine"], out, "portrait", "fast", 2, True, 1, None
                )
                self.assertEqual([os.path.basename(f) for f in files], ["001.mp4", "003.mp4"])
                with VideoFileClip(files[0]) as clip:
                    self.assertEqual(clip.size, [544, 960])
                    self.assertAlmostEqual(clip.fps, 24, delta=0.1)
                    self.assertAlmostEqual(clip.duration, 49 / 24, delta=0.1)
                self.assertEqual(os.listdir(os.path.join(comfy, "output")), [])

                calls.clear()
                wan_clips.generate(["reef", "jellyfish", "submarine"], out, "portrait", "fast", 2, True, 1, None)
                self.assertEqual(calls, ["clip_002"])  # finished clips are not redone

            zip_path = os.path.join(tmp, "ai_clips.zip")
            wan_clips.make_zip(files, zip_path)
            self.assertTrue(os.path.getsize(zip_path) > 0)

    def test_read_prompts(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fp:
            fp.write("# comment\n\nfirst\n  second  \n")
        try:
            self.assertEqual(wan_clips.read_prompts(fp.name), ["first", "second"])
        finally:
            os.remove(fp.name)


if __name__ == "__main__":
    unittest.main()
