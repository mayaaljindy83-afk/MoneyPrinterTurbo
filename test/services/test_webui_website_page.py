import json
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import llm
from app.services.marketing import pipeline, website
from app.services.presenter import profiles, studio

REAL_READ = website.read_website
PAGE = os.path.join(os.path.dirname(__file__), "..", "..", "webui", "pages", "2_AI_Website_Video.py")

SITE = {
    "url": "https://qai-vo.com/products/academic", "final_url": "https://qai-vo.com/products/academic",
    "mode": "service", "lang": "en", "dir": "ltr",
    "brand": {"name": "QAI-vo", "colors": ["#0f172a", "#10b981"], "logo": ""},
    "service": {"name": "Academic Suite", "name_id": "t1", "description": "Tools for researchers.",
                "description_id": "t2"},
    "benefits": [{"id": "t3", "text": "Paper & DOI Finder", "detail": "", "screenshot": "card1"}],
    "cta": [{"id": "t4", "text": "Upgrade", "href": "", "screenshot": ""}],
    "services": [], "assets": [], "source_urls": [],
    "texts": [{"id": "t1", "kind": "h1", "text": "Academic Suite"},
              {"id": "t2", "kind": "description", "text": "Tools for researchers."},
              {"id": "t3", "kind": "card_title", "text": "Paper & DOI Finder"},
              {"id": "t4", "kind": "cta", "text": "Upgrade"}],
    "screenshots": [{"id": "hero", "kind": "hero", "path": "screenshots/hero.png", "text": "Academic Suite"},
                    {"id": "card1", "kind": "card", "path": "screenshots/card1.png", "text": "Paper & DOI Finder"}],
}


def fake_read(url, out_dir, **kwargs):
    os.makedirs(os.path.join(out_dir, "screenshots"), exist_ok=True)
    for shot in SITE["screenshots"]:
        Image.new("RGB", (400, 250), (240, 240, 240)).save(os.path.join(out_dir, shot["path"]))
    with open(os.path.join(out_dir, "website.json"), "w", encoding="utf-8") as fp:
        json.dump(SITE, fp)
    return website.load_website(out_dir)


class TestWebsitePage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [
            mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp}),
            mock.patch.dict(config.app, {"presenters_dir": "", "kaggle_api_token": "", "default_presenter": "",
                                         "website_source_folder": "", "website_public_url": ""}),
            mock.patch.object(config, "save_config"),
            mock.patch.object(llm, "_generate_response", side_effect=RuntimeError("no llm here")),
            mock.patch.object(website, "read_website", fake_read),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    real_read = staticmethod(REAL_READ)

    def _presenter(self):
        src = os.path.join(self.tmp, "p.png")
        Image.new("RGB", (400, 600), (200, 170, 160)).save(src)
        profiles.save_presenter(profiles.Presenter(name="QAI"), [src])

    def _app(self):
        app = AppTest.from_file(PAGE, default_timeout=60)
        app.run()
        self.assertFalse(app.exception, app.exception)
        return app

    def test_asks_for_a_presenter_first(self):
        app = self._app()
        self.assertIn("اعملي المقدّمة أول شي من صفحة Presenter Video.", [i.value for i in app.info])

    def test_analyze_shows_storyboard_and_locks_full_video(self):
        self._presenter()
        app = self._app()
        app.text_input[0].set_value("https://qai-vo.com/products/academic")
        next(b for b in app.button if b.label == "حلّلي الموقع").click().run()
        self.assertFalse(app.exception, app.exception)
        projects = pipeline.list_projects()
        self.assertEqual(len(projects), 1)
        project = pipeline.load_project(projects[0])
        self.assertEqual(project["presenter"], "QAI")
        self.assertEqual(project["aspect"], "16:9")  # Facebook
        headers = [h.value for h in app.subheader]
        self.assertIn("الخدمة: Academic Suite", headers)
        self.assertTrue(any("ما قدرنا نوصل للذكاء الاصطناعي" in w.value for w in app.warning))
        preview = next(b for b in app.button if b.label == "ولّدي معاينة 10 ثواني")
        full = next(b for b in app.button if b.label == "ولّدي الفيديو الكامل")
        self.assertTrue(preview.disabled)  # no Kaggle token yet
        self.assertTrue(full.disabled)  # preview first

    def test_interrupted_preview_offers_continue(self):
        self._presenter()
        project = pipeline.analyze("https://qai-vo.com/products/academic", "en", 30, "youtube", "leads",
                                   presenter="QAI", read=fake_read, generate=lambda p: "not json")
        job_id = pipeline.create_job(project["project_id"], "preview")
        studio.set_status(job_id, "running", "sending the job to Kaggle", kaggle="running")
        with mock.patch.dict(config.app, {"kaggle_api_token": "abc"}):
            app = self._app()
        self.assertTrue(any(b.label.startswith("كمّلي") and not b.disabled for b in app.button))
        self.assertTrue(any("sending the job to Kaggle" in c.value for c in app.code))

    def test_local_source_mode(self):
        from test.services.test_marketing_website_source import make_project

        self._presenter()
        project_dir = os.path.join(self.tmp, "qai-vo-launch")
        make_project(project_dir)
        app = self._app()
        app.radio(key="web_source_mode").set_value("كود الموقع على اللابتوب").run()
        app.text_input[0].set_value(project_dir).run()
        self.assertTrue(any("2 صفحة عامة" in c.value for c in app.caption))
        self.assertEqual(app.selectbox[0].value, "/products/academic")
        app.text_input[1].set_value("https://qai-vo.com")
        with mock.patch.object(website, "read_website", self.real_read):
            next(b for b in app.button if b.label == "حلّلي الموقع").click().run()
        self.assertFalse(app.exception, app.exception)
        project = pipeline.load_project(pipeline.list_projects()[0])
        self.assertEqual(project["source"], {"folder": project_dir, "route": "/products/academic"})
        self.assertEqual(project["url"], "https://qai-vo.com/products/academic")
        self.assertEqual(config.app["website_source_folder"], project_dir)


if __name__ == "__main__":
    unittest.main()
