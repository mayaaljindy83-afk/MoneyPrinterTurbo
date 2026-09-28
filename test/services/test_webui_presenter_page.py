import os
import tempfile
import unittest
from unittest import mock

from PIL import Image
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import llm
from app.services.presenter import profiles, studio
from app.services.presenter import package as job_package

PAGE = os.path.join(os.path.dirname(__file__), "..", "..", "webui", "pages", "1_Presenter_Video.py")


class TestPresenterPage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [
            mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp}),
            mock.patch.dict(config.app, {"presenters_dir": "", "kaggle_api_token": "", "branding_dir":
                                         os.path.join(self.tmp, "branding")}),
            mock.patch.object(config, "save_config"),
            mock.patch.object(llm, "_generate_response", side_effect=RuntimeError("no llm here")),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def _app(self):
        app = AppTest.from_file(PAGE, default_timeout=60)
        app.run()
        self.assertFalse(app.exception, app.exception)
        return app

    def test_empty_page_asks_for_a_presenter(self):
        app = self._app()
        self.assertIn("اعملي مقدّمة بالخطوة 2 أول.", [i.value for i in app.info])
        app.radio(key="presenter_ui_lang").set_value("English").run()
        self.assertIn("Create a presenter in step 2 first.", [i.value for i in app.info])

    def test_plan_and_edit_shots(self):
        src = os.path.join(self.tmp, "p.png")
        Image.new("RGB", (400, 600), (200, 170, 160)).save(src)
        profiles.save_presenter(profiles.Presenter(name="Lina"), [src])
        app = self._app()
        app.text_input[1].set_value("Our website")  # [0] is the Kaggle token
        app.text_area[1].set_value("مرحبا بكم. " * 30)  # [0] is the presenter look
        plan_button = next(b for b in app.button if b.label == "خطّطي اللقطات")
        plan_button.click().run()
        self.assertFalse(app.exception, app.exception)
        jobs = job_package.list_jobs()
        self.assertEqual(len(jobs), 1)
        plan = job_package.load_plan(jobs[0])
        self.assertEqual(plan["presenter"], "Lina")
        self.assertGreater(len(plan["shots"]), 1)
        # Without a Kaggle token the render button is disabled.
        run_button = next(b for b in app.button if b.label.startswith("ولّدي على Kaggle"))
        self.assertTrue(run_button.disabled)

    def test_status_is_shown(self):
        src = os.path.join(self.tmp, "p.png")
        Image.new("RGB", (400, 600), (200, 170, 160)).save(src)
        profiles.save_presenter(profiles.Presenter(name="Lina"), [src])
        plan = studio.plan_video("t", "Hello there. " * 20, "en-US", "16:9", "Lina")
        studio.set_status(plan["job_id"], "running", "sending the job to Kaggle", kaggle="running")
        app = self._app()
        self.assertTrue(any("sending the job to Kaggle" in c.value for c in app.code))

    def test_interrupted_job_offers_continue(self):
        src = os.path.join(self.tmp, "p.png")
        Image.new("RGB", (400, 600), (200, 170, 160)).save(src)
        profiles.save_presenter(profiles.Presenter(name="Lina"), [src])
        plan = studio.plan_video("t", "Hello there. " * 20, "en-US", "16:9", "Lina")
        studio.set_status(plan["job_id"], "running", "sending the job to Kaggle", kaggle="running")
        app = self._app()
        self.assertTrue(any(plan["job_id"] in w.value for w in app.warning))
        self.assertTrue(any(b.label.startswith("كمّلي") for b in app.button))

    def test_runpod_is_selectable_and_enables_render(self):
        src = os.path.join(self.tmp, "p.png")
        Image.new("RGB", (400, 600), (200, 170, 160)).save(src)
        profiles.save_presenter(profiles.Presenter(name="Lina"), [src])
        studio.plan_video("t", "Hello there. " * 20, "en-US", "16:9", "Lina")
        with mock.patch.dict(config.app, {"presenter_cloud": "runpod", "runpod_api_key": "k",
                                          "runpod_endpoint_id": "ep"}):
            app = self._app()
            labels = [t.label for t in app.text_input]
            self.assertIn("مفتاح RunPod API", labels)
            self.assertNotIn("مفتاح Kaggle API", labels)
            run_button = next(b for b in app.button if b.label.startswith("ولّدي على RunPod"))
            self.assertFalse(run_button.disabled)  # RunPod set up, no Kaggle token needed


if __name__ == "__main__":
    unittest.main()
