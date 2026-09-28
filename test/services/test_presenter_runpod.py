"""RunPod Serverless: the endpoint handler and the laptop agent, talking through a fake RunPod API.

No request leaves the machine (RunPod bills every GPU second): ``FakeRunPod`` plays the
RunPod HTTP API and runs ``runpod/handler.py`` in-process with a fake worker.
"""

import base64
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock

from app.config import config
from app.services.presenter import cloud, runpod_agent, studio
from app.services.presenter import package as job_package
from app.utils import utils

HANDLER_PATH = os.path.join(utils.root_dir(), "runpod", "handler.py")


def load_handler():
    spec = importlib.util.spec_from_file_location("mpt_runpod_handler", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_worker(shot_bytes: dict, fail: str = ""):
    """Stands in for kaggle/presenter_worker.py: writes the shots the job asks for."""
    calls = []

    def main(argv):
        args, rest = {}, list(argv)
        while rest:
            name = rest.pop(0)
            args[name] = rest.pop(0) if rest and not rest[0].startswith("--") else True
        calls.append(args)
        with open(os.path.join(args["--job"], "job.json"), encoding="utf-8") as fp:
            job = json.load(fp)
        out = args["--out"]
        os.makedirs(os.path.join(out, "shots"), exist_ok=True)
        with open(os.path.join(out, "worker.log"), "a", encoding="utf-8") as fp:
            fp.write("[10:00:00] [s01] TALK 6.5s — أهلاً\n")
        if fail:
            raise SystemExit(fail)
        done = []
        for shot in job["shots"]:
            path = os.path.join(out, "shots", f"{shot['id']}.mp4")
            if not os.path.exists(path) and shot["id"] in shot_bytes:
                with open(path, "wb") as fp:
                    fp.write(shot_bytes[shot["id"]])
            if os.path.exists(path):
                done.append(shot["id"])
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fp:
            json.dump({"job_id": job["job_id"], "total": len(job["shots"]), "done": len(done),
                       "remaining": [s["id"] for s in job["shots"] if s["id"] not in done],
                       "complete": len(done) == len(job["shots"])}, fp)
        return 0

    return types.SimpleNamespace(main=main, calls=calls)


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeRunPod:
    """The RunPod v2 endpoint API, backed by the real handler."""

    def __init__(self, handler, key="key", laptop_off=False):
        self.handler = handler
        self.key = key
        self.jobs = {}
        self.requests = []
        self.cancelled = []
        self.laptop_off = laptop_off  # RunPod forgot the finished request before the laptop asked

    def request(self, method, url, json=None, headers=None, timeout=None):
        path = url.split("/v2/", 1)[1].split("/", 1)[1]
        self.requests.append((method, path, (json or {}).get("input", {}).get("mode")))
        if headers.get("Authorization") != f"Bearer {self.key}":
            return FakeResponse(401, {"error": "unauthorized"})
        if path == "health":
            return FakeResponse(200, {"workers": {"idle": 1, "running": 0}})
        if path in ("run", "runsync"):
            job_id = f"req-{len(self.jobs) + 1}"
            output = self.handler.handler({"id": job_id, "input": json["input"]})
            self.jobs[job_id] = output
            if path == "runsync":
                return FakeResponse(200, {"id": job_id, "status": "COMPLETED", "output": output})
            return FakeResponse(200, {"id": job_id, "status": "IN_QUEUE"})
        if path.startswith("cancel/"):
            self.cancelled.append(path.split("/", 1)[1])
            return FakeResponse(200, {"id": path.split("/", 1)[1], "status": "CANCELLED"})
        if path.startswith("status/"):
            job_id = path.split("/", 1)[1]
            if self.laptop_off or job_id not in self.jobs:
                return FakeResponse(404, {"error": "not found"})
            output = self.jobs[job_id]
            return FakeResponse(200, {"id": job_id, "status": "COMPLETED", "output": output,
                                      "delayTime": 42000, "executionTime": 300000})
        return FakeResponse(404, {})

    def modes(self):
        return [mode for _, _, mode in self.requests if mode]


class RunPodCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.volume = os.path.join(self.tmp, "volume")
        os.makedirs(self.volume)
        self.patches = [
            mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": os.path.join(self.tmp, "data")}),
            mock.patch.dict(config.app, {"presenter_cloud": "runpod", "runpod_api_key": "key",
                                         "runpod_endpoint_id": "ep1", "kaggle_api_token": ""}),
        ]
        for patch in self.patches:
            patch.start()
        self.handler = load_handler()
        self.handler.VOLUME = self.volume
        self.shot = os.urandom(9 * 1024 + 7)
        self.worker = fake_worker({"s01": self.shot, "s02": b"second" * 100})
        self.modules = mock.patch.dict(sys.modules, {"presenter_worker": self.worker})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        for patch in reversed(self.patches):
            patch.stop()

    def make_job(self, job_id="20260928-100000-preview", shots=("s01", "s02"), extra=b""):
        package = os.path.join(job_package.job_dir(job_id), "package")
        os.makedirs(os.path.join(package, "audio"), exist_ok=True)
        with open(os.path.join(package, "job.json"), "w", encoding="utf-8") as fp:
            json.dump({"job_id": job_id, "kind": "video",
                       "shots": [{"id": s, "type": "TALK", "duration": 6.5, "audio": f"audio/{s}.mp3",
                                  "text": "أهلاً وسهلاً"} for s in shots]}, fp, ensure_ascii=False)
        for s in shots:
            with open(os.path.join(package, "audio", f"{s}.mp3"), "wb") as fp:
                fp.write(b"ID3" + os.urandom(64))
        if extra:
            with open(os.path.join(package, "big.bin"), "wb") as fp:
                fp.write(extra)
        return job_id

    def agent(self, fake, **kwargs):
        return runpod_agent.RunPodAgent(api_key=kwargs.pop("api_key", "key"), endpoint_id="ep1", session=fake,
                                        poll_seconds=0, log=lambda m: None, sleep=lambda s: None, **kwargs)


class TestHandler(RunPodCase):
    def _zip(self, files: dict) -> str:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        return base64.b64encode(buffer.getvalue()).decode()

    def test_render_keeps_results_on_the_volume_and_fetches_in_pieces(self):
        job = json.dumps({"job_id": "j1", "shots": [{"id": "s01"}]})
        result = self.handler.handler({"id": "r1", "input": {"mode": "render", "job_id": "j1",
                                                             "package": self._zip({"job.json": job}), "fresh": True}})
        self.assertFalse(result.get("error"))
        self.assertTrue(result["summary"]["complete"])
        self.assertIn({"path": "shots/s01.mp4", "size": len(self.shot)}, result["files"])
        args = self.worker.calls[0]
        self.assertEqual(args["--cache"], os.path.join(self.volume, "models"))  # models download once
        self.assertIs(args["--skip-setup"], True)  # ComfyUI is already in the image
        self.assertEqual(os.environ["MPT_MODEL_STORE"], os.path.join(self.volume, "models"))
        pieces, offset = b"", 0
        while offset < len(self.shot):
            answer = self.handler.handler({"input": {"mode": "fetch", "job_id": "j1", "path": "shots/s01.mp4",
                                                     "offset": offset, "length": 4000}})
            data = base64.b64decode(answer["data"])
            pieces += data
            offset += len(data)
        self.assertEqual(pieces, self.shot)
        self.assertEqual(self.handler.handler({"input": {"mode": "cleanup", "job_id": "j1"}})["deleted"], True)
        self.assertFalse(os.path.exists(os.path.join(self.volume, "jobs", "j1")))

    def test_unsafe_paths_are_refused(self):
        self.handler.handler({"input": {"mode": "render", "job_id": "j2",
                                        "package": self._zip({"job.json": '{"job_id":"j2","shots":[]}'})}})
        answer = self.handler.handler({"input": {"mode": "fetch", "job_id": "j2", "path": "../../../etc/passwd"}})
        self.assertIn("outside", answer["error"])
        self.assertIn("bad job_id", self.handler.handler({"input": {"mode": "list", "job_id": "../x"}})["error"])
        evil = self._zip({"job.json": "{}", "../escape.txt": "x"})
        answer = self.handler.handler({"input": {"mode": "render", "job_id": "j3", "package": evil}})
        self.assertIn("outside", answer["error"])
        self.assertFalse(os.path.exists(os.path.join(self.volume, "jobs", "j3", "escape.txt")))

    def test_without_network_volume_the_worker_disk_is_used(self):
        self.handler.VOLUME = os.path.join(self.tmp, "no-volume")
        self.handler.LOCAL_STORE = os.path.join(self.tmp, "worker-disk")
        job = json.dumps({"job_id": "j9", "shots": [{"id": "s01"}]})
        result = self.handler.handler({"input": {"mode": "render", "job_id": "j9", "fresh": True,
                                                 "package": self._zip({"job.json": job})}})
        self.assertTrue(result["summary"]["complete"])
        self.assertFalse(result["persistent"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "worker-disk", "jobs", "j9", "output", "shots", "s01.mp4")))
        self.assertEqual(self.worker.calls[-1]["--cache"], os.path.join(self.tmp, "worker-disk", "models"))

    def test_worker_failure_before_any_shot_reports_the_log(self):
        sys.modules["presenter_worker"] = fake_worker({}, fail="ComfyUI did not start")
        answer = self.handler.handler({"input": {"mode": "render", "job_id": "j4",
                                                 "package": self._zip({"job.json": '{"job_id":"j4","shots":[{"id":"s01"}]}'})}})
        self.assertIn("ComfyUI did not start", answer["error"])
        self.assertIn("أهلاً", answer["error"])  # Arabic log lines survive


class TestAgent(RunPodCase):
    def test_run_downloads_every_shot_then_cleans_the_volume(self):
        job_id = self.make_job()
        fake = FakeRunPod(self.handler)
        with mock.patch.object(runpod_agent, "PIECE", 4000):
            summary = self.agent(fake).run_job(job_id, on_status=lambda s: None)
        self.assertTrue(summary["complete"])
        output = os.path.join(job_package.job_dir(job_id), "output")
        with open(os.path.join(output, "shots", "s01.mp4"), "rb") as fp:
            self.assertEqual(fp.read(), self.shot)
        self.assertEqual(fake.modes().count("render"), 1)
        self.assertGreaterEqual(fake.modes().count("fetch"), 3)  # the shot came in several pieces
        self.assertEqual(fake.modes()[-1], "cleanup")
        self.assertFalse(os.path.exists(os.path.join(self.volume, "jobs", job_id)))
        with open(os.path.join(job_package.job_dir(job_id), "runpod.json"), encoding="utf-8") as fp:
            self.assertEqual(json.load(fp)["request"], "req-1")
        self.assertTrue(cloud.has_run(job_id))
        with open(os.path.join(job_package.job_dir(job_id), "runpod.json"), encoding="utf-8") as fp:
            timings = json.load(fp)["timings"]  # for benchmark.json
        self.assertEqual((timings["queue_seconds"], timings["execution_seconds"]), (42.0, 300.0))
        self.assertIn("download_seconds", timings)

    def test_cancel_stops_the_request(self):
        job_id = self.make_job()
        fake = FakeRunPod(self.handler)
        agent = self.agent(fake)
        self.assertFalse(agent.cancel(job_id))  # nothing sent yet
        agent.submit(job_id, fresh=True)
        studio.set_status(job_id, "running", kaggle="running")
        self.assertTrue(studio.can_cancel(job_id))
        self.assertTrue(studio.cancel_cloud_run(job_id, agent=agent))
        self.assertEqual(fake.cancelled, ["req-1"])

    def test_package_never_uploads_previous_shots(self):
        job_id = self.make_job()
        previous = os.path.join(job_package.job_dir(job_id), "package", "previous", "shots")
        os.makedirs(previous)
        with open(os.path.join(previous, "s01.mp4"), "wb") as fp:
            fp.write(b"old")
        names = zipfile.ZipFile(io.BytesIO(base64.b64decode(self.agent(FakeRunPod(self.handler)).zip_package(job_id)))).namelist()
        self.assertIn("job.json", names)
        self.assertFalse([n for n in names if n.startswith("previous")])

    def test_continue_after_laptop_was_off_fetches_without_a_new_run(self):
        job_id = self.make_job()
        fake = FakeRunPod(self.handler)
        agent = self.agent(fake)
        agent.submit(job_id, fresh=True)  # the run finished while the laptop was off...
        fake.laptop_off = True  # ...and RunPod has already forgotten the request
        summary = agent.resume(job_id, max_runs=1)
        self.assertTrue(summary["complete"])
        self.assertEqual(fake.modes().count("render"), 1)  # no second GPU run
        self.assertTrue(os.path.isfile(os.path.join(job_package.job_dir(job_id), "output", "shots", "s02.mp4")))

    def test_failed_run_raises_with_the_worker_log(self):
        sys.modules["presenter_worker"] = fake_worker({}, fail="download failed after 8 attempts")
        job_id = self.make_job()
        states = []
        with self.assertRaises(runpod_agent.RunPodError) as ctx:
            self.agent(FakeRunPod(self.handler)).run_job(job_id, max_runs=1, on_status=states.append)
        self.assertIn("download failed", str(ctx.exception))
        self.assertEqual(states[-1], "error")  # "Render" is offered again, not a fetch-only Continue

    def test_run_error_is_shown_even_when_nothing_can_be_fetched(self):
        job_id = self.make_job()
        self.handler.render = lambda job, progress=None: {"error": "CUDA out of memory"}
        self.handler.list_job = lambda job_id: (_ for _ in ()).throw(self.handler.HandlerError("disk gone"))
        messages, states = [], []
        agent = runpod_agent.RunPodAgent(api_key="key", endpoint_id="ep1", session=FakeRunPod(self.handler),
                                         poll_seconds=0, log=messages.append, sleep=lambda s: None)
        with self.assertRaises(runpod_agent.RunPodError) as ctx:
            agent.run_job(job_id, max_runs=1, on_status=states.append)
        self.assertIn("CUDA out of memory", str(ctx.exception))
        self.assertTrue(any(m.startswith("RunPod error:") and "CUDA" in m for m in messages))
        self.assertEqual(states[-1], "error")

    def test_results_gone_with_the_worker_offers_render_again(self):
        job_id = self.make_job()
        self.handler.VOLUME = os.path.join(self.tmp, "no-volume")
        self.handler.LOCAL_STORE = os.path.join(self.tmp, "worker-disk")
        fake = FakeRunPod(self.handler)
        agent = self.agent(fake)
        agent.submit(job_id, fresh=True)
        shutil.rmtree(os.path.join(self.tmp, "worker-disk"))  # the worker scaled down: its disk is gone
        states = []
        with self.assertRaises(runpod_agent.ResultsLost) as ctx:
            agent.resume(job_id, max_runs=1, on_status=states.append)
        self.assertIn("Render", str(ctx.exception))
        self.assertEqual(states[-1], "error")  # studio.kaggle_run_failed -> Render, not an endless Continue

    def test_partial_run_is_reported_not_rerun_by_continue(self):
        sys.modules["presenter_worker"] = fake_worker({"s01": b"one"})
        job_id = self.make_job()
        fake = FakeRunPod(self.handler)
        summary = self.agent(fake).run_job(job_id, max_runs=1)
        self.assertEqual(summary["remaining"], ["s02"])
        self.assertNotIn("cleanup", fake.modes())  # keep the finished shot for the next run

    def test_too_big_package_is_refused_before_sending(self):
        job_id = self.make_job(extra=os.urandom(runpod_agent.MAX_PACKAGE + 1000))
        fake = FakeRunPod(self.handler)
        with self.assertRaises(runpod_agent.RunPodError) as ctx:
            self.agent(fake).run_job(job_id)
        self.assertIn("MB", str(ctx.exception))
        self.assertEqual(fake.requests, [])

    def test_wrong_key(self):
        with self.assertRaises(runpod_agent.RunPodError) as ctx:
            self.agent(FakeRunPod(self.handler), api_key="wrong").check()
        self.assertIn("rejected", str(ctx.exception))
        self.assertIn("1 idle", self.agent(FakeRunPod(self.handler)).check())


class TestChoosingTheService(RunPodCase):
    def test_setting_picks_the_agent(self):
        self.assertIsInstance(cloud.make_agent(), runpod_agent.RunPodAgent)
        self.assertTrue(cloud.ready())
        with mock.patch.dict(config.app, {"presenter_cloud": "kaggle"}):
            self.assertEqual(cloud.make_agent(token="t").__class__.__name__, "KaggleAgent")
            self.assertFalse(cloud.ready())  # no Kaggle token
        with mock.patch.dict(config.app, {"runpod_endpoint_id": ""}):
            self.assertFalse(cloud.ready())

    def test_continue_uses_the_service_that_ran_the_job(self):
        job_id = self.make_job()
        with open(os.path.join(job_package.job_dir(job_id), "kaggle.json"), "w", encoding="utf-8") as fp:
            json.dump({"kernel": "me/mpt-presenter-x"}, fp)
        # Switched to RunPod after a Kaggle run: Continue must fetch from Kaggle, not start a RunPod run.
        self.assertEqual(cloud.make_agent(token="t", job_id=job_id).__class__.__name__, "KaggleAgent")
        self.assertIsInstance(cloud.make_agent(token="t"), runpod_agent.RunPodAgent)

    def test_studio_render_and_continue_through_runpod(self):
        job_id = self.make_job(shots=("s01",))
        job_package.save_plan(job_id, {"job_id": job_id, "kind": "video"})
        fake = FakeRunPod(self.handler)
        agent = self.agent(fake)
        with mock.patch.object(studio, "render_final", lambda job, options=None: "final.mp4"):
            studio._run_on_kaggle(job_id, "", {}, agent=agent)
        self.assertEqual(studio.read_status(job_id)["state"], "rendered")
        self.assertTrue(studio.has_kaggle_run(job_id))
        self.assertEqual(fake.modes().count("render"), 1)


if __name__ == "__main__":
    unittest.main()
