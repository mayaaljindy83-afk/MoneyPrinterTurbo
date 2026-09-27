"""Read the website from its project folder on this laptop (no upload, no internet).

For a built Next.js project (``.next``) or a static export (``out``, ``dist``,
``build``), the built pages are served from 127.0.0.1 and read with the same
Playwright reader as a live site, with every request to another host blocked.
Nothing from the project is executed on this computer (no npm, no Node server,
no database); only the already-built HTML/CSS/JS runs inside the headless
browser, like when the page is opened in a browser.

From the source files it also takes the brand colours (CSS variables), the
logo file, and the list of components. Private areas (admin, dashboard, app,
login pages) are not offered, and folders such as ``payments`` or ``uploads``
are never used as video assets.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import os
import re
import threading
import urllib.parse

from loguru import logger

from app.services.marketing import website
from app.services.marketing.website import WebsiteError

STATIC_DIRS = ("out", "dist", "build")
PRIVATE_ROUTE = re.compile(r"^/(admin|dashboard|app|api)(/|$)|login|logout|password|verify|forbidden|"
                           r"_not-found|_global-error|/account|/checkout|receipt", re.IGNORECASE)
PRIVATE_ASSET_DIR = re.compile(r"(^|[\\/])(payments?|uploads?|receipts?|private|admin|secrets?)([\\/]|$)",
                               re.IGNORECASE)
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".svg")
SKIP_DIRS = {"node_modules", ".git", ".next", "out", "dist", "build", ".vercel", ".turbo", "coverage"}


def discover(folder: str) -> dict:
    """What kind of project this is and which public pages it has."""
    folder = os.path.abspath(os.path.expanduser(folder or ""))
    if not os.path.isdir(folder):
        raise WebsiteError(f"Folder not found: {folder}")
    next_app = os.path.join(folder, ".next", "server", "app")
    next_pages = os.path.join(folder, ".next", "server", "pages")
    for pages_dir, kind in ((next_app, "next"), (next_pages, "next")):
        if os.path.isdir(pages_dir):
            return {"kind": kind, "folder": folder, "pages_dir": pages_dir,
                    "static_dir": os.path.join(folder, ".next", "static"),
                    "public_dir": os.path.join(folder, "public"), "routes": _routes(pages_dir)}
    for name in STATIC_DIRS:
        root = os.path.join(folder, name)
        if os.path.isfile(os.path.join(root, "index.html")):
            return {"kind": "static", "folder": folder, "pages_dir": root, "static_dir": root,
                    "public_dir": root, "routes": _routes(root)}
    if os.path.isfile(os.path.join(folder, "index.html")):
        return {"kind": "static", "folder": folder, "pages_dir": folder, "static_dir": folder,
                "public_dir": folder, "routes": _routes(folder)}
    raise WebsiteError("No built website found in this folder. Build it once on this laptop "
                       "(for Next.js: npm run build), then try again.")


def _routes(pages_dir: str) -> list[str]:
    routes = []
    for root, dirs, files in os.walk(pages_dir):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
        for name in files:
            if not name.endswith(".html"):
                continue
            rel = os.path.relpath(os.path.join(root, name), pages_dir).replace(os.sep, "/")[:-5]
            route = "/" if rel == "index" else "/" + re.sub(r"/index$", "", rel)
            if not PRIVATE_ROUTE.search(route):
                routes.append(route)
    return sorted(set(routes), key=lambda r: (r.count("/"), r))


def _inside(base: str, path: str) -> str:
    """``path`` joined to ``base`` if it stays inside it (no ``..`` escapes), else ""."""
    base = os.path.realpath(base)
    target = os.path.realpath(os.path.join(base, path))
    return target if target == base or target.startswith(base + os.sep) else ""


class _SiteHandler(http.server.BaseHTTPRequestHandler):
    project: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802 (http.server naming)
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        project = self.project
        target = ""
        if path.startswith("/_next/static/"):
            target = _inside(project["static_dir"], path[len("/_next/static/"):])
        elif path.startswith("/_next/image"):
            source = urllib.parse.parse_qs(parsed.query).get("url", [""])[0]
            target = _inside(project["public_dir"], urllib.parse.unquote(source).lstrip("/"))
        elif not path.startswith("/api/"):
            rel = path.strip("/")
            for candidate in ((rel or "index") + ".html", os.path.join(rel, "index.html"), rel):
                found = _inside(project["pages_dir"], candidate)
                if found and os.path.isfile(found):
                    target = found
                    break
            if not target and rel:
                target = _inside(project["public_dir"], rel)
        if target and os.path.isfile(target):
            self._send(target)
        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, path: str):
        import mimetypes

        kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".html"):
            kind = "text/html; charset=utf-8"
        with open(path, "rb") as fp:
            body = fp.read()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def serve(project: dict):
    """Serve the built site on 127.0.0.1 (only this computer can reach it)."""
    handler = type("Handler", (_SiteHandler,), {"project": project})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- source files
_CSS_VAR = re.compile(r"--([\w-]+)\s*:\s*(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})\b")
_PREFERRED_VARS = ("accent", "primary", "brand", "emerald", "indigo", "navy", "ink", "bg", "background")


def _walk(folder: str, extensions: tuple[str, ...]):
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.lower().endswith(extensions) and not name.startswith(".env"):
                yield os.path.join(root, name)


def brand_colors(folder: str) -> list[str]:
    """Hex colours from CSS variables (``:root { --accent: #10b981 }``), brand-like names first."""
    found: dict[str, str] = {}
    for path in _walk(folder, (".css", ".scss")):
        try:
            with open(path, encoding="utf-8", errors="replace") as fp:
                text = fp.read(400_000)
        except OSError:
            continue
        for name, value in _CSS_VAR.findall(text):
            value = value.lower()
            if len(value) == 4:
                value = "#" + "".join(c * 2 for c in value[1:])
            found.setdefault(name, value)
    ranked = sorted(found.items(), key=lambda item: next(
        (i for i, key in enumerate(_PREFERRED_VARS) if key in item[0]), len(_PREFERRED_VARS)))
    colors = []
    for _, value in ranked:
        if value not in colors:
            colors.append(value)
    return colors[:8]


def find_logo(folder: str) -> str:
    candidates = [p for p in _walk(folder, IMAGE_EXT)
                  if "logo" in os.path.basename(p).lower() and not PRIVATE_ASSET_DIR.search(os.path.relpath(p, folder))]
    candidates.sort(key=lambda p: (p.lower().endswith(".svg"), "public" not in p.lower(), len(p)))
    return candidates[0] if candidates else ""


def components(folder: str) -> list[str]:
    names = []
    for path in _walk(os.path.join(folder, "src") if os.path.isdir(os.path.join(folder, "src")) else folder,
                      (".tsx", ".jsx", ".vue", ".svelte")):
        name = os.path.splitext(os.path.basename(path))[0]
        if name[:1].isupper() and name not in names:
            names.append(name)
    return sorted(names)[:200]


def _logo_png(logo: str, out_dir: str) -> str:
    """The logo file as PNG (SVGs are drawn by the browser)."""
    target = os.path.join(out_dir, "screenshots", "logo.png")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not logo.lower().endswith(".svg"):
        from PIL import Image

        Image.open(logo).convert("RGBA").save(target)
        return target
    from playwright.sync_api import sync_playwright

    website.browsers_path()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            with open(logo, encoding="utf-8", errors="replace") as fp:
                svg = fp.read()
            page.set_content(f"<body style='margin:0;background:transparent'>{svg}</body>")
            page.query_selector("svg").screenshot(path=target, omit_background=True)
        finally:
            browser.close()
    return target


def read_local_site(folder: str, route: str = "/", out_dir: str | None = None, locale: str = "",
                    public_url: str = "", portrait: bool = True) -> dict:
    """Read one page of the local project like a website, plus brand data from its source files."""
    project = discover(folder)
    route = "/" + (route or "/").strip().lstrip("/")
    if PRIVATE_ROUTE.search(route):
        raise WebsiteError(f"{route} is a private page (admin, account or login); choose a public page.")
    out_dir = out_dir or website.project_dir_for("local:" + project["folder"] + route)
    with serve(project) as base:
        data = website.read_website(base + route, out_dir, portrait=portrait, check_robots=False,
                                    locale=locale, offline=True)
    colors = brand_colors(project["folder"])
    if colors:
        data["brand"]["colors"] = colors + [c for c in data["brand"]["colors"] if c not in colors]
    logo = find_logo(project["folder"])
    if logo:
        try:
            _logo_png(logo, out_dir)
            data["screenshots"] = [s for s in data["screenshots"] if s["id"] != "logo"]
            data["screenshots"].append({"id": "logo", "kind": "logo", "path": "screenshots/logo.png",
                                        "text": data["brand"]["name"]})
            data["brand"]["logo"] = "screenshots/logo.png"
        except Exception as exc:  # the page's own logo screenshot stays
            logger.warning(f"logo file {logo} could not be used: {exc}")
    if public_url:
        public = website.normalize_url(public_url).rstrip("/")
        data["final_url"] = public + route
        data["source_urls"] = [data["final_url"]]
        for item in data.get("services", []):
            item["url"] = public + urllib.parse.urlparse(item["url"]).path
    data["source"] = {"kind": "local", "folder": project["folder"], "route": route, "build": project["kind"],
                      "components": components(project["folder"])}
    data.pop("_dir", None)
    with open(os.path.join(out_dir, "website.json"), "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    data["_dir"] = out_dir
    return data


def reader(folder: str, route: str, public_url: str = ""):
    """A ``read_website``-compatible function for ``pipeline.analyze``."""
    return functools.partial(_read_for_pipeline, folder, route, public_url)


def _read_for_pipeline(folder, route, public_url, url, out_dir, locale="", **kwargs):
    return read_local_site(folder, route, out_dir, locale=locale, public_url=public_url)
