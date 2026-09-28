"""Regression: a render stopped on Windows with

    [WinError 32] The process cannot access the file because it is being used by another
    process: '...status.json.tmp' -> '...status.json'

because the page (or an antivirus) had status.json open while the render replaced it.
"""

import builtins
import os
import tempfile
import threading
import unittest
from unittest import mock

from app.services.presenter import studio
from app.services.presenter import package as job_package


def _locked(times):
    """os.replace that fails like Windows for the first ``times`` calls."""
    real = os.replace
    calls = {"n": 0}

    def replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= times:
            raise PermissionError(32, "The process cannot access the file because it is being used by "
                                      "another process", src)
        return real(src, dst)

    return replace, calls


class TestWindowsFileLocks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()
        job_package.save_plan("job1", {"job_id": "job1", "shots": []})
        studio.set_status("job1", "running", "بدأ التشغيل على Kaggle", kaggle="running")

    def tearDown(self):
        self.env.stop()

    def test_status_write_waits_for_a_locked_file(self):
        replace, calls = _locked(3)
        with mock.patch.object(studio.os, "replace", replace), mock.patch.object(studio.time, "sleep"):
            studio.set_status("job1", message="first frame ready", kaggle="running")
        self.assertEqual(calls["n"], 4)
        status = studio.read_status("job1")
        self.assertEqual(status["state"], "running")
        self.assertEqual([line.split(" ", 1)[1] for line in status["log"]],
                         ["بدأ التشغيل على Kaggle", "first frame ready"])
        self.assertFalse([f for f in os.listdir(job_package.job_dir("job1")) if f.endswith(".tmp")])

    def test_file_locked_for_good_is_written_in_place(self):
        replace, _ = _locked(10_000)
        with mock.patch.object(studio.os, "replace", replace), mock.patch.object(studio.time, "sleep"):
            studio.set_status("job1", message="still going")
        self.assertIn("still going", studio.read_status("job1")["log"][-1])

    def test_locked_read_does_not_wipe_the_log(self):
        real_open = builtins.open
        fails = {"n": 2}

        def open_(path, *args, **kwargs):
            if str(path).endswith("status.json") and fails["n"]:
                fails["n"] -= 1
                raise PermissionError(32, "being used by another process", path)
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", open_), mock.patch.object(studio.time, "sleep"):
            studio.set_status("job1", message="downloading the video model")
        log = studio.read_status("job1")["log"]
        self.assertEqual(len(log), 2)  # the earlier line is still there
        self.assertEqual(studio.read_status("job1")["state"], "running")

    def test_page_and_render_writing_together(self):
        def writer(tag):
            for i in range(25):
                studio.set_status("job1", message=f"{tag} {i}")

        threads = [threading.Thread(target=writer, args=(tag,)) for tag in ("render", "page")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(studio.read_status("job1")["log"]), 51)


if __name__ == "__main__":
    unittest.main()
