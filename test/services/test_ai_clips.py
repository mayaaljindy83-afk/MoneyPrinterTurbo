import os
import tempfile
import unittest
from unittest import mock

from app.models.schema import VideoParams
from app.services import ai_clips, llm, task


class TestMixing(unittest.TestCase):
    def test_clips_are_spread_in_order(self):
        stock = [f"s{i}" for i in range(10)]
        mixed = ai_clips.mix_ai_clips(stock, ["a1", "a2"])
        self.assertEqual(len(mixed), 12)
        self.assertEqual([c for c in mixed if c.startswith("s")], stock)
        self.assertLess(mixed.index("a1"), mixed.index("a2"))
        # a1 lands in the first half, a2 in the second half of the timeline.
        self.assertLess(mixed.index("a1"), 6)
        self.assertGreater(mixed.index("a2"), 6)

    def test_more_ai_clips_than_stock(self):
        mixed = ai_clips.mix_ai_clips(["s0"], ["a1", "a2", "a3"])
        self.assertEqual(sorted(mixed), ["a1", "a2", "a3", "s0"])
        self.assertEqual([c for c in mixed if c.startswith("a")], ["a1", "a2", "a3"])

    def test_edge_cases(self):
        self.assertEqual(ai_clips.mix_ai_clips(["s0"], []), ["s0"])
        self.assertEqual(ai_clips.mix_ai_clips([], ["a1"]), ["a1"])


class TestFolders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patch = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_list_uses_natural_order_and_video_files_only(self):
        folder = os.path.join(ai_clips.ai_clips_root(), "ocean")
        os.makedirs(folder)
        for name in ("10.mp4", "2.mp4", "1.mp4", "notes.txt", ".hidden.mp4"):
            open(os.path.join(folder, name), "wb").close()
        names = [os.path.basename(f) for f in ai_clips.list_ai_clips(folder)]
        self.assertEqual(names, ["1.mp4", "2.mp4", "10.mp4"])

    def test_resolve_rejects_paths_outside_root(self):
        os.makedirs(os.path.join(ai_clips.ai_clips_root(), "ocean"))
        self.assertTrue(ai_clips.resolve_ai_clips_folder("ocean").endswith("ocean"))
        for bad in ("", "../", "../../etc", "missing"):
            with self.assertRaises(ValueError):
                ai_clips.resolve_ai_clips_folder(bad)

    def test_pipeline_mix_skips_missing_folder(self):
        params = VideoParams(video_subject="x", ai_clips_folder="missing")
        self.assertEqual(task._mix_in_ai_clips(params, ["s0"]), ["s0"])

    def test_pipeline_mix_adds_clips(self):
        folder = os.path.join(ai_clips.ai_clips_root(), "ocean")
        os.makedirs(folder)
        open(os.path.join(folder, "001.mp4"), "wb").close()
        params = VideoParams(video_subject="x", ai_clips_folder="ocean")
        mixed = task._mix_in_ai_clips(params, ["s0", "s1"])
        self.assertEqual(len(mixed), 3)
        self.assertTrue(any(c.endswith("001.mp4") for c in mixed))


class TestPrompts(unittest.TestCase):
    def test_one_prompt_per_paragraph_in_order(self):
        seen = []

        def fake(prompt, app_config=None):
            seen.append(prompt)
            paragraph = prompt.split("this part of the video:\n")[1].split("\n")[0]
            return f"Prompt: **Cinematic shot of {paragraph}**, slow dolly-in."

        script = "Deep ocean trench.\n\nGlowing fish.\n\nA robot submarine."
        with mock.patch.object(llm, "_generate_response", fake):
            prompts = ai_clips.build_scene_prompts("Ocean", script)
        self.assertEqual(
            prompts,
            [
                "Cinematic shot of Deep ocean trench., slow dolly-in.",
                "Cinematic shot of Glowing fish., slow dolly-in.",
                "Cinematic shot of A robot submarine., slow dolly-in.",
            ],
        )
        self.assertIn("clearly different", seen[1])
        text = ai_clips.prompts_to_text(prompts)
        self.assertEqual(text.count("\n"), 3)

    def test_llm_error_is_raised(self):
        with mock.patch.object(llm, "_generate_response", return_value="Error: down"):
            with self.assertRaises(RuntimeError):
                ai_clips.build_scene_prompts("x", "one paragraph")


if __name__ == "__main__":
    unittest.main()
