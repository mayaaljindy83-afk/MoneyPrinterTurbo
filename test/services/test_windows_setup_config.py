import importlib.util
import os
import shutil
import tempfile
import unittest

import toml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    "setup_config", os.path.join(ROOT, "windows", "setup_config.py")
)
setup_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_config)


class TestWindowsSetupConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = os.path.join(self.tmp, "config.toml")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_setup(self, *args):
        setup_config.main(["--config", self.config, *args])
        return toml.load(self.config)

    def test_first_run_creates_laptop_config(self):
        data_dir = os.path.join(self.tmp, "Data Folder")
        cfg = self.run_setup(
            "--data-dir", data_dir, "--pexels-key", "px", "--gemini-key", "gm",
            "--resolution", "720p",
        )
        app = cfg["app"]
        self.assertEqual(app["storage_dir"], setup_config.normalize_path(data_dir))
        self.assertTrue(os.path.isdir(os.path.join(data_dir, "music")))
        self.assertEqual(app["pexels_api_keys"], ["px"])
        self.assertEqual(app["llm_provider"], "ollama")
        self.assertEqual(app["ollama_model_name"], "gemma3:4b")
        self.assertEqual(app["llm_fallback_provider"], "gemini")
        self.assertEqual(app["gemini_model_name"], "gemini-2.5-flash")
        self.assertEqual(app["video_resolution"], "720p")
        self.assertTrue(app["fast_clip_preparation"])
        self.assertEqual(cfg["ui"]["voice_name"], "ar-SA-HamedNeural-Male")
        self.assertNotIn("\\", app["storage_dir"])

    def test_rerun_keeps_existing_keys_and_user_choices(self):
        self.run_setup("--pexels-key", "px", "--gemini-key", "gm")
        cfg = toml.load(self.config)
        cfg["ui"]["voice_name"] = "ar-SY-LaithNeural-Male"
        with open(self.config, "w", encoding="utf-8") as fp:
            toml.dump(cfg, fp)
        cfg = self.run_setup("--pexels-key", "", "--gemini-key", "")
        self.assertEqual(cfg["app"]["pexels_api_keys"], ["px"])
        self.assertEqual(cfg["app"]["gemini_api_key"], "gm")
        self.assertEqual(cfg["ui"]["voice_name"], "ar-SY-LaithNeural-Male")

    def test_dash_removes_keys(self):
        self.run_setup("--pexels-key", "px", "--gemini-key", "gm")
        cfg = self.run_setup("--pexels-key", "-", "--gemini-key", "-")
        self.assertEqual(cfg["app"]["pexels_api_keys"], [])
        self.assertEqual(cfg["app"]["llm_fallback_provider"], "")

    def test_config_toml_is_gitignored(self):
        with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as fp:
            self.assertIn("/config.toml", fp.read().splitlines())


if __name__ == "__main__":
    unittest.main()
