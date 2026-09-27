import http.client
import json
import os
import tempfile
import unittest
import urllib.parse

from PIL import Image

from app.services.marketing import pipeline, website, website_source

# A built page like Next.js produces: static HTML + /_next/static CSS, language from localStorage.
PAGE = """<!doctype html><html lang="en" dir="ltr"><head><meta charset="utf-8">
<title>Academic Suite · QAI-vo</title><link rel="stylesheet" href="/_next/static/css/app.css"></head>
<body><header><a href="/"><img src="/brand/qai-vo-logo.png" alt="QAI-vo" width="120" height="40"></a>
<select id="lang"><option value="en">English</option><option value="ar">العربية</option></select></header>
<main><h1 id="title">Academic Suite</h1><p id="desc">Everything you need for research.</p>
<div class="cards"><div class="card"><h3 id="c1">Paper &amp; DOI Finder</h3><p>Find sources with DOI links.</p></div></div>
<button class="btn-primary" id="cta">Upgrade</button>
<img src="https://tracker.example.com/pixel.png" alt="">
</main>
<script>
 const ar = {title: "الحقيبة الأكاديمية", desc: "كل ما تحتاجه للبحث العلمي.", c1: "باحث الأوراق وروابط DOI",
             cta: "ترقية الآن"};
 function apply(l) { if (l !== "ar") return; document.documentElement.lang = "ar"; document.documentElement.dir = "rtl";
   for (const k in ar) document.getElementById(k).textContent = ar[k]; }
 apply(localStorage.getItem("qaivo_locale"));
 document.getElementById("lang").onchange = (e) => { localStorage.setItem("qaivo_locale", e.target.value); apply(e.target.value); };
</script></body></html>"""

CSS = """body{margin:0;font-family:sans-serif;background:#f8fafc;color:#0f172a}
.cards{display:flex;padding:20px}.card{width:320px;padding:16px;border:1px solid #ddd;border-radius:14px;background:#fff}
.btn-primary{background:#10b981;color:#fff;border:0;padding:12px 24px;border-radius:10px}"""


def make_project(root):
    app = os.path.join(root, ".next", "server", "app")
    for rel, body in {"index.html": PAGE, "products/academic.html": PAGE, "admin/users.html": "<h1>admin</h1>",
                      "login.html": "<h1>login</h1>", "_not-found.html": "x"}.items():
        os.makedirs(os.path.dirname(os.path.join(app, rel)), exist_ok=True)
        with open(os.path.join(app, rel), "w", encoding="utf-8") as fp:
            fp.write(body)
    os.makedirs(os.path.join(root, ".next", "static", "css"))
    with open(os.path.join(root, ".next", "static", "css", "app.css"), "w", encoding="utf-8") as fp:
        fp.write(CSS)
    for rel, color in {"public/brand/qai-vo-logo.png": (16, 185, 129), "public/payments/whish-qr.jpg": (0, 0, 0)}.items():
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        Image.new("RGB", (240, 80), color).save(os.path.join(root, rel))
    os.makedirs(os.path.join(root, "src", "app"))
    with open(os.path.join(root, "src", "app", "globals.css"), "w", encoding="utf-8") as fp:
        fp.write(":root { --navy: #0f172a; --emerald: #10b981; --indigo: #6366f1; --accent: #10b981; --bg: #f8fafc; }")
    os.makedirs(os.path.join(root, "src", "components"))
    open(os.path.join(root, "src", "components", "ServiceCard.tsx"), "w").close()
    with open(os.path.join(root, ".env"), "w", encoding="utf-8") as fp:
        fp.write("STRIPE_SECRET_KEY=sk_live_never_read_me")
    with open(os.path.join(root, "secret.txt"), "w", encoding="utf-8") as fp:
        fp.write("top secret")


class TestLocalSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.project = os.path.join(cls.root, "qai-vo-launch")
        make_project(cls.project)

    def test_discover_lists_only_public_pages(self):
        info = website_source.discover(self.project)
        self.assertEqual(info["kind"], "next")
        self.assertEqual(info["routes"], ["/", "/products/academic"])

    def test_brand_logo_and_components_from_source(self):
        self.assertEqual(website_source.brand_colors(self.project)[:2], ["#10b981", "#6366f1"])
        self.assertTrue(website_source.find_logo(self.project).endswith("qai-vo-logo.png"))
        self.assertEqual(website_source.components(self.project), ["ServiceCard"])

    def test_server_stays_inside_the_build(self):
        info = website_source.discover(self.project)
        with website_source.serve(info) as base:
            host = urllib.parse.urlparse(base)
            conn = http.client.HTTPConnection(host.hostname, host.port)
            for path, status in (("/products/academic", 200), ("/_next/static/css/app.css", 200),
                                 ("/brand/qai-vo-logo.png", 200), ("/_next/static/../../../secret.txt", 404),
                                 ("/../secret.txt", 404), ("/..%2f..%2fsecret.txt", 404), ("/.env", 404),
                                 ("/api/users", 404)):
                conn.request("GET", path)
                response = conn.getresponse()
                response.read()
                self.assertEqual(response.status, status, path)

    def test_arabic_page_read_offline(self):
        out = tempfile.mkdtemp()
        data = website_source.read_local_site(self.project, "/products/academic", out, locale="ar",
                                              public_url="https://qai-vo.com")
        self.assertEqual(data["service"]["name"], "الحقيبة الأكاديمية")  # the Arabic version of the page
        self.assertEqual(data["dir"], "rtl")
        self.assertEqual(data["final_url"], "https://qai-vo.com/products/academic")
        self.assertEqual(data["brand"]["colors"][0], "#10b981")
        self.assertEqual(data["source"]["components"], ["ServiceCard"])
        self.assertIn("ترقية الآن", [c["text"] for c in data["cta"]])
        logo = Image.open(os.path.join(out, "screenshots", "logo.png")).convert("RGB")
        self.assertEqual(logo.getpixel((5, 5)), (16, 185, 129))  # the real logo file, not the QR code
        saved = json.load(open(os.path.join(out, "website.json"), encoding="utf-8"))
        self.assertEqual(saved["service"]["name"], "الحقيبة الأكاديمية")
        self.assertNotIn("sk_live", json.dumps(saved))

    def test_english_page(self):
        data = website_source.read_local_site(self.project, "/products/academic", tempfile.mkdtemp(), locale="en",
                                              portrait=False)
        self.assertEqual(data["service"]["name"], "Academic Suite")

    def test_private_pages_are_refused(self):
        with self.assertRaises(website.WebsiteError):
            website_source.read_local_site(self.project, "/admin/users", tempfile.mkdtemp())

    def test_folder_without_build(self):
        empty = tempfile.mkdtemp()
        with self.assertRaises(website.WebsiteError) as ctx:
            website_source.discover(empty)
        self.assertIn("npm run build", str(ctx.exception))

    def test_pipeline_analyze_from_local_folder(self):
        tmp = tempfile.mkdtemp()
        from unittest import mock

        with mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": tmp}):
            project = pipeline.analyze("https://qai-vo.com", "ar", 20, "facebook", "subscriptions", presenter="QAI",
                                       source_folder=self.project, route="/products/academic",
                                       generate=lambda p: "not json")
        self.assertEqual(project["url"], "https://qai-vo.com/products/academic")
        self.assertEqual(project["source"]["route"], "/products/academic")
        self.assertEqual(project["plan"]["service_name"], "الحقيبة الأكاديمية")
        self.assertIn("qai-vo.com", project["plan"]["scenes"][-1]["voiceover"])  # public address, not 127.0.0.1


if __name__ == "__main__":
    unittest.main()


class TestLocalSourceScreenshotTimeout(unittest.TestCase):
    def test_arabic_local_page_is_analysed_when_screenshots_hang(self):
        from unittest import mock

        from playwright.sync_api import Page
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        project = os.path.join(tempfile.mkdtemp(), "qai-vo-launch")
        make_project(project)

        def hang(*args, **kwargs):
            raise PlaywrightTimeout("Page.screenshot: Timeout 30000ms exceeded.")

        out = tempfile.mkdtemp()
        with mock.patch.object(Page, "screenshot", autospec=True, side_effect=hang):
            data = website_source.read_local_site(project, "/products/academic", out, locale="ar",
                                                  public_url="https://qai-vo.com")
        self.assertEqual(data["service"]["name"], "الحقيبة الأكاديمية")
        layers = {s["id"]: s for s in data["screenshots"] if s.get("layer")}
        self.assertEqual(layers["card1"]["text"], "باحث الأوراق وروابط DOI")  # the Arabic component
        self.assertIn("cta1", layers)
        self.assertEqual(data["brand"]["colors"][0], "#10b981")
        self.assertTrue(os.path.isfile(os.path.join(out, "screenshots", "logo.png")))  # from the logo file
