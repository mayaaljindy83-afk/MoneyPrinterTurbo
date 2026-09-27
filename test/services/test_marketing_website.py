import functools
import http.server
import os
import tempfile
import threading
import unittest
from unittest import mock

from PIL import Image

from app.services.marketing import website

SERVICE_PAGE = """<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>التدقيق الأكاديمي | QAI-VO</title>
<meta name="description" content="تدقيق لغوي وأكاديمي لأبحاثك خلال 48 ساعة.">
<meta property="og:site_name" content="QAI-VO">
<style>
 body{font-family:sans-serif;margin:0;background:#0F172A;color:#fff}
 header{background:#0F172A;padding:12px;display:flex;justify-content:space-between}
 .logo{width:120px;height:40px}
 .cards{display:flex;gap:20px;padding:20px}
 .card{width:300px;background:#1e293b;padding:16px;border-radius:12px}
 .btn-primary{background:#10B981;color:#fff;padding:12px 24px;border:0;border-radius:8px}
</style></head><body>
<header><img class="logo" alt="QAI-VO logo" src="logo.png"><nav><a href="/about.html">من نحن</a></nav></header>
<main>
<h1>خدمة التدقيق الأكاديمي</h1>
<p>نراجع بحثك لغوياً وأكاديمياً ونضمن سلامة المراجع والتنسيق.</p>
<ul><li>تدقيق لغوي كامل للنص العربي والإنجليزي</li><li>مراجعة التوثيق حسب APA و MLA</li></ul>
<div class="cards">
 <div class="card"><h3>سرعة التسليم</h3><p>نسلّم خلال 48 ساعة فقط.</p></div>
 <div class="card"><h3>خبراء متخصصون</h3><p>محررون حاصلون على الدكتوراه.</p></div>
</div>
<button class="btn-primary">اشترك الآن</button>
</main>
<footer><a href="/privacy.html">سياسة الخصوصية</a></footer>
</body></html>"""

HOME_PAGE = """<!doctype html><html lang="en"><head><title>QAI-VO - AI for research</title></head><body>
<nav><a href="/services/editing.html">Academic Editing</a><a href="/services/finance.html">Financial Analysis</a>
<a href="/login.html">Login</a><a href="/contact.html">Contact</a></nav>
<h1>AI tools for academics</h1>
<div style="width:300px"><h3>Legal Assistant</h3><p>Draft contracts faster with our assistant.</p>
<a href="/services/legal.html">Learn more</a></div>
</body></html>"""

LOGIN_PAGE = """<!doctype html><html><body><form><input name=u><input type=password name=p>
<button class="btn">Sign in</button></form></body></html>"""

ROBOTS = "User-agent: *\nDisallow: /private/\n"


class LocalSite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.root, "services"))
        os.makedirs(os.path.join(cls.root, "private"))
        files = {"service.html": SERVICE_PAGE, "index.html": HOME_PAGE, "login.html": LOGIN_PAGE,
                 "robots.txt": ROBOTS, "private/page.html": SERVICE_PAGE}
        for name, body in files.items():
            with open(os.path.join(cls.root, name), "w", encoding="utf-8") as fp:
                fp.write(body)
        Image.new("RGB", (240, 80), (16, 185, 129)).save(os.path.join(cls.root, "logo.png"))
        handler = functools.partial(_QuietHandler, directory=cls.root)
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.out = tempfile.mkdtemp()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class TestServicePage(LocalSite):
    def test_arabic_service_page(self):
        data = website.read_website(self.base + "/service.html", self.out)
        self.assertEqual(data["mode"], "service")
        self.assertEqual(data["service"]["name"], "خدمة التدقيق الأكاديمي")
        self.assertEqual(data["dir"], "rtl")
        # The visible intro under the heading (in the page's language), not the <meta> tag.
        self.assertEqual(data["service"]["description"], "نراجع بحثك لغوياً وأكاديمياً ونضمن سلامة المراجع والتنسيق.")
        self.assertEqual(data["brand"]["name"], "QAI-VO")
        self.assertIn("#10b981", data["brand"]["colors"])
        benefit_texts = [b["text"] for b in data["benefits"]]
        self.assertIn("سرعة التسليم", benefit_texts)
        self.assertIn("مراجعة التوثيق حسب APA و MLA", benefit_texts)
        self.assertEqual(data["cta"][0]["text"], "اشترك الآن")
        # Every benefit / CTA points at a numbered fact.
        ids = {t["id"] for t in data["texts"]}
        for item in data["benefits"] + data["cta"]:
            self.assertIn(item["id"], ids)
        # Real screenshots of the page and its parts exist.
        kinds = {s["kind"] for s in data["screenshots"]}
        self.assertTrue({"hero", "page", "card", "cta", "logo", "mobile"} <= kinds, kinds)
        for shot in data["screenshots"]:
            image = Image.open(os.path.join(self.out, shot["path"]))
            self.assertGreater(image.width, 10)
        card = next(s for s in data["screenshots"] if s["kind"] == "card")
        # Cards are the real components drawn alone: 2x resolution, transparent margin for shadows.
        self.assertTrue(card.get("layer"))
        layer = Image.open(os.path.join(self.out, card["path"]))
        self.assertEqual(layer.width, (332 + 48) * 2)  # 300 + padding, plus the 24 px margin, at 2x
        self.assertEqual(layer.getpixel((2, 2))[3], 0)  # transparent around the card
        self.assertEqual(layer.getpixel((layer.width // 2, layer.height // 2))[:3], (30, 41, 59))  # its own colour
        kinds = [s["kind"] for s in data["screenshots"] if s.get("layer")]
        self.assertEqual(sorted(set(kinds)), ["card", "cta", "description", "heading", "logo"])
        # Saved and loadable.
        self.assertEqual(website.load_website(self.out)["service"]["name"], data["service"]["name"])
        # Footer / navigation text is not a marketing fact.
        self.assertNotIn("سياسة الخصوصية", [t["text"] for t in data["texts"]])

    def test_homepage_lists_main_services(self):
        data = website.read_website(self.base + "/", self.out, portrait=False)
        self.assertEqual(data["mode"], "homepage")
        names = [s["name"] for s in data["services"]]
        self.assertIn("Academic Editing", names)
        self.assertIn("Learn more", names)  # card link to the legal service
        self.assertNotIn("Login", names)
        self.assertNotIn("Contact", names)

    def test_robots_txt_is_respected(self):
        with self.assertRaises(website.WebsiteError):
            website.read_website(self.base + "/private/page.html", self.out)

    def test_login_page_is_not_bypassed(self):
        with self.assertRaises(website.WebsiteError) as ctx:
            website.read_website(self.base + "/login.html", self.out)
        self.assertIn("login", str(ctx.exception).lower())

    def test_missing_page(self):
        with self.assertRaises(website.WebsiteError):
            website.read_website(self.base + "/nope.html", self.out)


class TestHelpers(unittest.TestCase):
    def test_normalize_url(self):
        self.assertEqual(website.normalize_url("qai-vo.com"), "https://qai-vo.com")
        self.assertEqual(website.normalize_url("https://a.com/x#top"), "https://a.com/x")
        for bad in ("file:///etc/passwd", "javascript:alert(1)", ""):
            with self.assertRaises(website.WebsiteError):
                website.normalize_url(bad)

    def test_homepage_detection(self):
        self.assertTrue(website.is_homepage("https://qai-vo.com/"))
        self.assertTrue(website.is_homepage("https://qai-vo.com/ar"))
        self.assertFalse(website.is_homepage("https://qai-vo.com/services/editing"))

    def test_browsers_live_on_the_data_drive(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": tmp}, clear=False):
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
            try:
                self.assertEqual(website.browsers_path(), os.path.join(tmp, "playwright-browsers"))
            finally:
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/pw-browsers" if os.path.isdir("/opt/pw-browsers") \
                    else os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")

    def test_facts_are_numbered_once(self):
        data = {"title": "A", "description": "B", "headings": [{"level": 1, "text": "A"}],
                "cards": [], "texts": [{"text": "C", "inList": True}], "ctas": [{"text": "Go"}]}
        texts, index = website.build_facts(data)
        self.assertEqual([t["id"] for t in texts], ["t1", "t2", "t3", "t4"])
        self.assertEqual(index["A"], "t1")


if __name__ == "__main__":
    unittest.main()


class TestScreenshotTimeouts(LocalSite):
    """A page whose screenshots hang (seen on Windows with a local Next.js build) must still be analysed."""

    def _timeout(self, *args, **kwargs):
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        raise PlaywrightTimeout("Page.screenshot: Timeout 30000ms exceeded.")

    def test_page_screenshots_time_out_components_still_rendered(self):
        from playwright.sync_api import Page

        with mock.patch.object(Page, "screenshot", autospec=True, side_effect=self._timeout) as shot:
            data = website.read_website(self.base + "/service.html", self.out, locale="ar")
        self.assertGreaterEqual(shot.call_count, 2)  # tried, then retried once per image
        ids = {s["id"] for s in data["screenshots"]}
        self.assertFalse({"hero", "page", "mobile"} & ids)
        self.assertTrue({"heading", "card1", "card2", "cta1", "logo"} <= ids)  # independent layers
        self.assertEqual(data["service"]["name"], "خدمة التدقيق الأكاديمي")
        self.assertIn("سرعة التسليم", [b["text"] for b in data["benefits"]])
        self.assertEqual(data["cta"][0]["text"], "اشترك الآن")
        self.assertIn("#10b981", data["brand"]["colors"])
        self.assertTrue(any("hero.png" in w for w in data["screenshot_warnings"]))
        self.assertIn("running_animations", data["render_diagnostics"])

    def test_no_image_at_all_still_completes(self):
        from playwright.sync_api import Locator, Page

        with mock.patch.object(Page, "screenshot", autospec=True, side_effect=self._timeout), \
                mock.patch.object(Locator, "screenshot", autospec=True, side_effect=self._timeout):
            data = website.read_website(self.base + "/service.html", self.out, portrait=False)
        self.assertEqual(data["screenshots"], [])
        self.assertEqual(data["service"]["name"], "خدمة التدقيق الأكاديمي")
        self.assertTrue(data["benefits"] and data["cta"] and data["texts"])

    def test_full_page_fails_then_viewport_retry_works(self):
        from playwright.sync_api import Page

        real = Page.screenshot

        def full_page_hangs(page, *args, **kwargs):
            if kwargs.get("full_page"):
                self._timeout()
            return real(page, *args, **kwargs)

        with mock.patch.object(Page, "screenshot", autospec=True, side_effect=full_page_hangs):
            data = website.read_website(self.base + "/service.html", self.out, portrait=False)
        ids = {s["id"] for s in data["screenshots"]}
        self.assertTrue({"hero", "page"} <= ids)  # page.png = the visible part
        self.assertTrue(any("only the visible part" in w for w in data["screenshot_warnings"]))
