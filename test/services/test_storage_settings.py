import os
import tempfile
import unittest
from unittest import mock

from app.config import config
from app.services import bgm
from app.utils import utils


class TestStorageRoot(unittest.TestCase):
    def test_default_is_inside_project(self):
        with mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": ""}), mock.patch.dict(
            config.app, {"storage_dir": ""}
        ):
            self.assertEqual(utils.storage_root(), os.path.join(utils.root_dir(), "storage"))

    def test_config_value_moves_all_storage(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"MPT_STORAGE_DIR": ""}
        ), mock.patch.dict(config.app, {"storage_dir": tmp}):
            self.assertEqual(utils.storage_root(), os.path.abspath(tmp))
            self.assertEqual(utils.storage_dir("cache_videos"), os.path.join(tmp, "cache_videos"))
            task = utils.task_dir("abc")
            self.assertTrue(os.path.isdir(task))
            self.assertTrue(task.startswith(os.path.abspath(tmp)))

    def test_environment_variable_wins(self):
        with tempfile.TemporaryDirectory() as env_dir, mock.patch.dict(
            os.environ, {"MPT_STORAGE_DIR": env_dir}
        ), mock.patch.dict(config.app, {"storage_dir": "/somewhere/else"}):
            self.assertEqual(utils.storage_root(), os.path.abspath(env_dir))


class TestUserMusicFolder(unittest.TestCase):
    def test_music_dir_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = os.path.join(tmp, "music")
            with mock.patch.dict(config.app, {"music_dir": music}):
                self.assertEqual(bgm.uploaded_bgm_dir(create=True), os.path.abspath(music))
                self.assertTrue(os.path.isdir(music))

    def test_prefer_user_music_replaces_builtin_songs(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = os.path.join(tmp, "calm piano.mp3")
            with open(track, "wb") as f:
                f.write(b"\x00" * 16)
            with mock.patch.dict(
                config.app, {"music_dir": tmp, "bgm_prefer_user_music": True}
            ):
                files = bgm.list_bgm_files()
                self.assertEqual([os.path.basename(f) for f in files], ["calm piano.mp3"])
                self.assertEqual(
                    os.path.realpath(bgm.resolve_bgm_file("calm piano.mp3")),
                    os.path.realpath(track),
                )
            with mock.patch.dict(
                config.app, {"music_dir": tmp, "bgm_prefer_user_music": False}
            ):
                self.assertGreater(len(bgm.list_bgm_files()), 1)

    def test_prefer_user_music_falls_back_when_folder_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            config.app, {"music_dir": tmp, "bgm_prefer_user_music": True}
        ):
            self.assertGreater(len(bgm.list_bgm_files()), 0)


if __name__ == "__main__":
    unittest.main()
