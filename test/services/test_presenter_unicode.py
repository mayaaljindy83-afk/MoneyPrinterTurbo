"""Regression: a finished Kaggle run crashed while downloading on Windows.

``kaggle.api.kaggle_api_extended.KaggleApi.kernels_output`` saves the run log
with ``open(outfile, "w")`` (no encoding). On Windows that is cp1252
("charmap"), which cannot encode the Arabic narration in the log:
UnicodeEncodeError right after status COMPLETE, although the videos had
already been downloaded.
"""

import builtins
import contextlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from app.services.presenter import kaggle_agent, studio
from app.services.presenter import package as job_package

ARABIC_LOG = ("[16:40:12] [s01] TALK 4.7s\n"
              "narration: أهلاً وسهلاً! هذا اختبار قصير للمقدّمة الافتراضية.\n"
              "[16:52:30] [s01] done in 11.3 min\n")


def _windows_open(file, mode="r", *args, **kwargs):
    """What a bare ``open()`` does on a Windows machine with the cp1252 code page."""
    if "b" not in mode and "encoding" not in kwargs and len(args) < 2:
        kwargs["encoding"] = "cp1252"
    return builtins.open(file, mode, *args, **kwargs)


def _kaggle_module():
    with mock.patch.dict(os.environ, {"KAGGLE_API_TOKEN": "dummy"}), \
            contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8")):
        from kaggle.api import kaggle_api_extended
    return kaggle_api_extended


class FakeClient:
    def __init__(self, response):
        self.kernels = SimpleNamespace(kernels_api_client=SimpleNamespace(
            list_kernel_session_output=lambda request: response))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestKaggleLogIsUtf8(unittest.TestCase):
    def setUp(self):
        self.module = _kaggle_module()
        self.original_open = self.module.__dict__.get("open")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        if self.original_open is None:
            self.module.__dict__.pop("open", None)
        else:
            self.module.open = self.original_open

    def _download(self):
        """Run the real kernels_output with a fake server: one video + an Arabic log."""
        response = SimpleNamespace(files=[SimpleNamespace(file_name="shots/s01.mp4", url="https://example/s01")],
                                   log=ARABIC_LOG, next_page_token="")
        api = self.module.KaggleApi()
        with mock.patch.object(api, "build_kaggle_client", return_value=FakeClient(response)), \
                mock.patch.object(self.module.requests, "get",
                                  return_value=SimpleNamespace(content=b"\x00\x00\x00\x18ftypmp42")):
            return api.kernels_output("mayaaljindy/mpt-presenter-test", self.tmp, force=True, quiet=True)

    def test_windows_code_page_reproduces_the_crash(self):
        self.module.open = _windows_open
        with self.assertRaises(UnicodeEncodeError) as ctx:
            self._download()
        self.assertIn("charmap", str(ctx.exception))
        # The video was already saved before the log crashed: nothing was lost on Kaggle's side.
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "shots", "s01.mp4")))

    def test_utf8_patch_saves_the_arabic_log(self):
        self.module.open = _windows_open  # even on a cp1252 machine...
        kaggle_agent.use_utf8_file_io(self.module)  # ...what KaggleAgent.api installs
        files, _ = self._download()
        log_path = next(f for f in files if f.endswith(".log"))
        with open(log_path, encoding="utf-8") as fp:
            self.assertEqual(fp.read(), ARABIC_LOG)  # Arabic kept exactly, not stripped or escaped
        with open(os.path.join(self.tmp, "shots", "s01.mp4"), "rb") as fp:
            self.assertEqual(fp.read(), b"\x00\x00\x00\x18ftypmp42")  # binary untouched

    def test_explicit_encodings_and_binary_modes_are_respected(self):
        path = os.path.join(self.tmp, "x.txt")
        with kaggle_agent._utf8_open(path, "w") as fp:
            self.assertEqual(fp.encoding.lower(), "utf-8")
        with kaggle_agent._utf8_open(path, "w", encoding="utf-16") as fp:
            self.assertEqual(fp.encoding, "utf-16")
        with kaggle_agent._utf8_open(path, "wb") as fp:
            self.assertFalse(hasattr(fp, "encoding"))


class TestArabicStateFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_status_keeps_arabic_as_utf8(self):
        job_package.save_plan("job1", {"job_id": "job1", "shots": [{"narration": "هذا اختبار قصير"}]})
        studio.set_status("job1", "running", "جاري التسجيل: هذا اختبار قصير")
        raw = open(studio.status_path("job1"), "rb").read()
        self.assertIn("هذا اختبار قصير".encode("utf-8"), raw)  # stored as real Arabic, not \\u escapes
        self.assertIn("هذا اختبار قصير", studio.read_status("job1")["log"][-1])
        plan_raw = open(os.path.join(job_package.job_dir("job1"), "plan.json"), "rb").read()
        self.assertIn("هذا اختبار قصير".encode("utf-8"), plan_raw)


class FakeApi:
    """A finished run: status COMPLETE, one shot of two made; must never be run again."""

    def __init__(self, shots_done):
        self.shots_done = shots_done
        self.pushed = []

    def get_config_value(self, name):
        return "mayaaljindy"

    def kernels_status(self, kernel):
        return SimpleNamespace(status=SimpleNamespace(name="COMPLETE"))

    def kernels_output(self, kernel, path, force, quiet, page_token):
        os.makedirs(os.path.join(path, "shots"), exist_ok=True)
        for shot in self.shots_done:
            open(os.path.join(path, "shots", f"{shot}.mp4"), "wb").write(b"video")
        with open(os.path.join(path, "worker.log"), "w", encoding="utf-8") as fp:
            fp.write(ARABIC_LOG)
        with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as fp:
            json.dump({"total": 2, "done": len(self.shots_done), "complete": len(self.shots_done) == 2,
                       "remaining": [s for s in ("s01", "s02") if s not in self.shots_done]}, fp)
        return [], None

    def kernels_push(self, folder):
        self.pushed.append(folder)

    def dataset_create_version(self, *args, **kwargs):
        self.pushed.append("dataset")


class TestContinueNeverStartsTheGpu(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()
        os.makedirs(os.path.join(job_package.job_dir("job1"), "package"))
        with open(os.path.join(job_package.job_dir("job1"), "package", "job.json"), "w", encoding="utf-8") as fp:
            json.dump({"job_id": "job1", "kind": "video", "shots": []}, fp)
        with open(os.path.join(job_package.job_dir("job1"), "kaggle.json"), "w", encoding="utf-8") as fp:
            json.dump({"dataset": "mayaaljindy/mpt-job-job1", "kernel": "mayaaljindy/mpt-presenter-job1"}, fp)

    def tearDown(self):
        self.env.stop()

    def _agent(self, api):
        return kaggle_agent.KaggleAgent(token="t", api=api, poll_seconds=0, log=lambda m: None, sleep=lambda s: None)

    def test_resume_with_one_run_only_downloads(self):
        api = FakeApi(["s01"])
        summary = self._agent(api).resume("job1", max_runs=1)
        self.assertEqual(summary["remaining"], ["s02"])
        self.assertEqual(api.pushed, [])  # no new GPU run
        log = open(os.path.join(job_package.job_dir("job1"), "output", "worker.log"), encoding="utf-8").read()
        self.assertIn("أهلاً وسهلاً", log)

    def test_failed_local_step_can_continue_after_the_error(self):
        job_package.save_plan("job1", {"job_id": "job1", "shots": []})
        studio.set_status("job1", "error", "ERROR: 'charmap' codec can't encode characters")
        self.assertTrue(studio.has_kaggle_run("job1"))
        self.assertTrue(studio.can_continue("job1"))
        job_package.save_plan("job2", {"job_id": "job2", "shots": []})
        studio.set_status("job2", "error", "ERROR: no Kaggle run yet")
        self.assertFalse(studio.can_continue("job2"))  # nothing to fetch


if __name__ == "__main__":
    unittest.main()
