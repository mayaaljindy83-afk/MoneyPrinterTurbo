"""Read one service page (or a homepage) with Playwright.

Everything the ad may say about the website must come from here: every text
gets an id (``t1``, ``t2``...) so the script can point at what it is based on,
and real screenshots (whole page, hero, cards, buttons, logo) are what the
video shows. Nothing is invented and no AI redraws the site.

Rules:
- robots.txt is respected; login pages and paywalls are not bypassed (a page
  that is only a login form is reported as such);
- only the given page is read (a homepage also lists its main services);
- page content is untrusted data: it is stored and later handed to the
  LLM as quoted data, never as instructions, and nothing on the page can run
  anything on this computer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.robotparser

import requests
from loguru import logger

from app.utils import utils

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/141.0 Safari/537.36 MoneyPrinterTurbo-WebsiteReader/1.0")
ROBOTS_AGENT = "MoneyPrinterTurbo-WebsiteReader"
DESKTOP = {"width": 1440, "height": 900}
MOBILE = {"width": 430, "height": 932}
MAX_PAGE_HEIGHT = 6000
MAX_CARDS = 8
MAX_CTAS = 3
MAX_TEXTS = 80


class WebsiteError(RuntimeError):
    pass


def browsers_path() -> str:
    """Playwright browsers live on the data drive (E:), never in the user profile on C:."""
    path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if not path:
        path = utils.storage_dir("playwright-browsers", create=True)
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = path
    return path


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if "://" not in url and re.match(r"^[a-z][a-z0-9+.-]*:(?!\d)", url, re.IGNORECASE):
        raise WebsiteError(f"not a website address: {url!r}")  # javascript:, data:, mailto:...
    if url and "://" not in url:
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise WebsiteError(f"not a website address: {url!r}")
    return urllib.parse.urlunparse(parsed._replace(fragment=""))


def is_homepage(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.strip("/").lower()
    return path in ("", "index.html", "index.php", "home", "ar", "en")


def robots_allowed(url: str, timeout: float = 10) -> bool:
    parsed = urllib.parse.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        logger.warning(f"robots.txt not reachable ({exc}); reading the page anyway")
        return True
    if response.status_code in (401, 403):
        return False  # the site refuses automated access
    if response.status_code >= 400:
        return True
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch(ROBOTS_AGENT, url)


def project_dir_for(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    host = re.sub(r"[^a-z0-9]+", "-", urllib.parse.urlparse(url).netloc.lower()).strip("-")
    return utils.storage_dir(os.path.join("websites", f"{host}-{digest}"), create=True)


# Runs inside the page. Returns plain data and marks elements for screenshots.
_EXTRACT_JS = r"""
() => {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    return r.width > 2 && r.height > 2 && st.visibility !== "hidden" && st.display !== "none"
      && parseFloat(st.opacity || "1") > 0.05;
  };
  const box = (el) => { const r = el.getBoundingClientRect();
    return {x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height}; };
  const rgb = (c) => {
    const m = (c || "").match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?/);
    if (!m || (m[4] !== undefined && parseFloat(m[4]) < 0.3)) return "";
    return "#" + [m[1], m[2], m[3]].map(v => (+v).toString(16).padStart(2, "0")).join("");
  };
  const meta = (sel) => { const e = document.querySelector(sel); return e ? clean(e.content) : ""; };

  const headings = [];
  document.querySelectorAll("h1,h2,h3").forEach(h => {
    if (visible(h) && clean(h.innerText)) headings.push({level: +h.tagName[1], text: clean(h.innerText).slice(0, 200)});
  });
  const texts = [];
  document.querySelectorAll("p,li,blockquote,dd,figcaption").forEach(el => {
    const t = clean(el.innerText);
    if (visible(el) && t.length >= 12 && t.length <= 500 && !el.closest("nav,footer,script,style"))
      texts.push({tag: el.tagName.toLowerCase(), text: t, inList: !!el.closest("ul,ol")});
  });

  // Cards: framed boxes (border, shadow or own background, rounded corners) of card size,
  // innermost first, so "Paper & DOI Finder" is one card, not the whole grid around it.
  const isCardBox = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 140 || r.width > 760 || r.height < 60 || r.height > 720 || !visible(el)) return false;
    if (el.closest("nav,footer,header")) return false;
    const st = getComputedStyle(el);
    const framed = parseFloat(st.borderTopWidth) > 0 || st.boxShadow !== "none" || rgb(st.backgroundColor) !== "";
    const t = clean(el.innerText);
    return framed && parseFloat(st.borderTopLeftRadius) >= 4 && t.length >= 12 && t.length <= 450;
  };
  const boxes = [...document.querySelectorAll("div,a,article,li,section")].filter(isCardBox);
  const leaves = boxes.filter(el => !boxes.some(other => other !== el && el.contains(other)));
  const cards = [];
  for (const el of leaves) {
    if (cards.length >= 16) break;
    let title = el.querySelector("h1,h2,h3,h4,h5,h6,strong,b");
    if (!title) title = [...el.querySelectorAll("p,span,div")].find(e => e.childElementCount === 0
        && clean(e.innerText) && parseInt(getComputedStyle(e).fontWeight) >= 600);
    const heading = clean(title ? title.innerText : el.innerText.split("\n")[0]).slice(0, 160);
    if (!heading) continue;
    el.setAttribute("data-mpt-card", String(cards.length));
    cards.push({heading, text: clean(el.innerText).slice(0, 400), box: box(el)});
  }

  const ctas = [];
  document.querySelectorAll("a,button,[role=button],input[type=submit]").forEach(el => {
    const t = clean(el.innerText || el.value);
    if (!visible(el) || t.length < 2 || t.length > 40 || el.closest("footer")) return;
    const st = getComputedStyle(el);
    const cls = (el.className && el.className.baseVal === undefined ? el.className : "") + " " + (el.id || "");
    const filled = rgb(st.backgroundColor) !== "" && rgb(st.backgroundColor) !== "#ffffff";
    const score = (el.tagName === "BUTTON" ? 2 : 0) + (/btn|button|cta|primary|signup|start|subscribe/i.test(cls) ? 3 : 0)
      + (filled ? 3 : 0) + (el.closest("nav,header") ? -1 : 0);
    if (score >= 3) ctas.push({text: t, href: el.href || "", score, el});
  });
  ctas.sort((a, b) => b.score - a.score);
  const ctaOut = [];
  for (const c of ctas) {
    if (ctaOut.length >= 6 || ctaOut.some(o => o.text === c.text)) continue;
    c.el.setAttribute("data-mpt-cta", String(ctaOut.length));
    ctaOut.push({text: c.text, href: c.href, box: box(c.el)});
  }

  let logo = null;
  for (const el of document.querySelectorAll("img,svg,a,[class*=logo],[id*=logo]")) {
    const attrs = [el.getAttribute("src"), el.getAttribute("alt"), el.getAttribute("class"), el.id,
                   el.getAttribute("aria-label")].join(" ");
    const r = el.getBoundingClientRect();
    if (/logo|brand/i.test(attrs) && visible(el) && r.width >= 20 && r.width <= 500 && r.height <= 250) {
      el.setAttribute("data-mpt-logo", "1");
      logo = {alt: clean(el.getAttribute("alt") || el.getAttribute("aria-label")), src: el.currentSrc || el.src || ""};
      break;
    }
  }

  if (!logo) {
    // Usual pattern: the first header link back to the homepage, holding an icon or image.
    const home = [...document.querySelectorAll("header a[href], nav a[href]")].find(a => {
      try { return new URL(a.href).pathname === "/" && a.querySelector("svg,img") && visible(a); } catch (e) { return false; }
    });
    if (home) { home.setAttribute("data-mpt-logo", "1"); logo = {alt: clean(home.innerText), src: ""}; }
  }
  const logoEl = document.querySelector("[data-mpt-logo]");

  const images = [];
  document.querySelectorAll("img").forEach(img => {
    const r = img.getBoundingClientRect();
    if (visible(img) && r.width >= 120 && r.height >= 80 && images.length < 20)
      images.push({src: img.currentSrc || img.src, alt: clean(img.alt).slice(0, 160), w: Math.round(r.width), h: Math.round(r.height)});
  });

  const links = [];
  const here = location.origin;
  document.querySelectorAll("a[href]").forEach(a => {
    const t = clean(a.innerText);
    try {
      const u = new URL(a.href, location.href);
      if (u.origin === here && t.length >= 3 && t.length <= 60 && visible(a) && !a.closest("footer"))
        links.push({text: t, url: u.href.split("#")[0], inNav: !!a.closest("nav,header"), inCard: !!a.closest("[data-mpt-card]")});
    } catch (e) {}
  });

  const colors = [];
  const push = (c) => { if (c && !colors.includes(c)) colors.push(c); };
  const firstCta = document.querySelector("[data-mpt-cta='0']");
  if (firstCta) push(rgb(getComputedStyle(firstCta).backgroundColor));
  const header = document.querySelector("header,nav");
  if (header) push(rgb(getComputedStyle(header).backgroundColor));
  const h1 = document.querySelector("h1");
  if (h1) push(rgb(getComputedStyle(h1).color));
  push(rgb(getComputedStyle(document.body).backgroundColor));
  const a = document.querySelector("a");
  if (a) push(rgb(getComputedStyle(a).color));
  if (logoEl) logoEl.querySelectorAll("*").forEach(e => { const c = rgb(getComputedStyle(e).color);
    if (c && c !== "#000000" && c !== "#ffffff") push(c); });

  return {
    title: clean(document.title), description: meta("meta[name=description]") || meta("meta[property='og:description']"),
    siteName: meta("meta[property='og:site_name']"), ogImage: meta("meta[property='og:image']"),
    lang: document.documentElement.lang || "", dir: document.documentElement.dir || getComputedStyle(document.body).direction,
    hasPassword: !!document.querySelector("input[type=password]"),
    bodyText: clean(document.body.innerText).length,
    headings, texts, cards, ctas: ctaOut, logo, logoBox: logoEl ? box(logoEl) : null, images, links, colors,
    pageHeight: document.documentElement.scrollHeight,
  };
}
"""


SCREENSHOT_TIMEOUT_MS = 20000
RETRY_TIMEOUT_MS = 10000

# Measures what can stall a screenshot: size, fixed layers, canvas/video, running animations,
# constant DOM changes, lazy images, heavy blur effects.
_DIAGNOSTICS_JS = r"""
async () => {
  const all = [...document.querySelectorAll("*")];
  let fixed = 0, backdrop = 0;
  for (const el of all) {
    const st = getComputedStyle(el);
    if (st.position === "fixed" || st.position === "sticky") fixed++;
    if ((st.backdropFilter && st.backdropFilter !== "none") || (st.webkitBackdropFilter && st.webkitBackdropFilter !== "none")) backdrop++;
  }
  let mutations = 0;
  const observer = new MutationObserver((list) => { mutations += list.length; });
  observer.observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
  await new Promise((r) => setTimeout(r, 1000));
  observer.disconnect();
  return {
    width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight,
    elements: all.length, fixed_or_sticky: fixed, backdrop_blur: backdrop,
    canvas: document.querySelectorAll("canvas").length, video: document.querySelectorAll("video").length,
    iframes: document.querySelectorAll("iframe").length,
    running_animations: document.getAnimations ? document.getAnimations().length : -1,
    dom_changes_per_second: mutations,
    lazy_images: document.querySelectorAll("img[loading=lazy]").length,
    unloaded_images: [...document.images].filter((i) => !i.complete).length,
  };
}
"""

# Stops what keeps a page repainting: CSS animations/transitions, the caret, smooth scrolling,
# videos, and loading overlays/spinners.
_CALM_CSS = """*, *::before, *::after { animation: none !important; transition: none !important;
  caret-color: transparent !important; scroll-behavior: auto !important; }"""
_CALM_JS = r"""
() => {
  document.querySelectorAll("video").forEach((v) => { try { v.pause(); } catch (e) {} });
  for (const el of document.querySelectorAll("[class*=loader],[class*=spinner],[class*=preloader],[class*=splash],"
                                               + "[id*=loader],[id*=spinner],[id*=preloader],[id*=splash]")) {
    const r = el.getBoundingClientRect();
    if (getComputedStyle(el).position === "fixed" || r.width * r.height > innerWidth * innerHeight * 0.5)
      el.style.setProperty("display", "none", "important");
  }
}
"""


# Collects each real component (cards, buttons, heading, intro text, logo) with everything needed
# to draw it alone: its HTML, the page's stylesheets, and the <html>/<body> classes (themes, fonts,
# direction). Runs in the live page after it has rendered.
_COMPONENTS_JS = r"""
(baseUrl) => {
  // Any CSS colour (rgb, oklab, oklch, color()...) to [r, g, b, a], converted by the browser itself.
  const paint = document.createElement("canvas").getContext("2d", {willReadFrequently: true});
  const rgba = (css) => {
    if (!css || css === "transparent") return [0, 0, 0, 0];
    paint.clearRect(0, 0, 1, 1);
    paint.fillStyle = "rgba(0,0,0,0)";
    paint.fillStyle = css;
    paint.fillRect(0, 0, 1, 1);
    const d = paint.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2], d[3] / 255];
  };
  const attrs = (el) => [...el.attributes].filter((a) => !/^on/i.test(a.name))
    .map((a) => `${a.name}="${a.value.replace(/&/g, "&amp;").replace(/"/g, "&quot;")}"`).join(" ");
  const h1 = document.querySelector("h1");
  if (h1) h1.setAttribute("data-mpt-heading", "1");
  let desc = null;
  if (h1) {
    for (let el = h1.nextElementSibling, i = 0; el && i < 4; el = el.nextElementSibling, i++)
      if (el.tagName === "P" && el.innerText.trim().length > 20) { desc = el; break; }
    if (!desc && h1.parentElement) desc = [...h1.parentElement.querySelectorAll("p")].find(
      (p) => p.innerText.trim().length > 20);
  }
  if (desc) desc.setAttribute("data-mpt-desc", "1");
  const styles = [...document.querySelectorAll("link[rel=stylesheet], style")].map((n) => n.outerHTML).join("\n");
  const html = document.documentElement, body = document.body;
  const pick = (sel, kind, id, text) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return null;
    // The colour a visitor sees behind the component: every (possibly translucent, glass-like)
    // ancestor background blended down to the first solid one (white if none).
    const layersUnder = [];
    for (let a = el.parentElement; a; a = a.parentElement) {
      const v = rgba(getComputedStyle(a).backgroundColor);
      const alpha = v[3];
      if (alpha > 0) layersUnder.push(v);
      if (alpha >= 0.99) break;
    }
    let seen = [255, 255, 255];
    for (const [r, g, b, alpha] of layersUnder.reverse()) seen = [r, g, b].map((c, i) => c * alpha + seen[i] * (1 - alpha));
    const bg = `rgb(${seen.map(Math.round).join(",")})`;
    // Glass-like parts (translucent background + blur) turn unreadable on a video background:
    // give the copy the colour it actually shows on the page (its colour over what is behind it).
    const clone = el.cloneNode(true);
    const own = rgba(getComputedStyle(el).backgroundColor);
    if (own[3] > 0 && own[3] < 0.9) {
      const a = own[3];
      const mix = [0, 1, 2].map((i) => Math.round(own[i] * a + seen[i] * (1 - a)));
      clone.style.setProperty("background-color", `rgb(${mix.join(",")})`, "important");
      clone.style.setProperty("backdrop-filter", "none", "important");
    }
    return {id, kind, text: text || el.innerText.trim().slice(0, 200), html: clone.outerHTML,
            width: Math.ceil(r.width), height: Math.ceil(r.height), inheritedBackground: bg,
            radius: getComputedStyle(el).borderTopLeftRadius};
  };
  const items = [];
  const add = (x) => { if (x) items.push(x); };
  add(pick("[data-mpt-heading]", "heading", "heading"));
  add(pick("[data-mpt-desc]", "description", "description"));
  document.querySelectorAll("[data-mpt-card]").forEach((el) => add(pick(
    `[data-mpt-card="${el.getAttribute("data-mpt-card")}"]`, "card", "card" + (+el.getAttribute("data-mpt-card") + 1))));
  document.querySelectorAll("[data-mpt-cta]").forEach((el) => add(pick(
    `[data-mpt-cta="${el.getAttribute("data-mpt-cta")}"]`, "cta", "cta" + (+el.getAttribute("data-mpt-cta") + 1))));
  add(pick("[data-mpt-logo]", "logo", "logo"));
  return {items, styles, base: baseUrl, htmlAttrs: attrs(html), bodyAttrs: attrs(body),
          dir: html.dir || getComputedStyle(body).direction, lang: html.lang || ""};
}
"""


def _component_page(bundle: dict, item: dict) -> str:
    """A static page (no scripts) that shows one component with the site's own CSS."""
    import html as html_lib

    background, radius = "transparent", "0"
    if item["kind"] in ("heading", "description"):  # text sits on the page colour, as on the site
        background, radius = item.get("inheritedBackground") or "transparent", "18px"
    inner = item["html"]
    if item["kind"] == "card" and item.get("inheritedBackground"):
        # A solid backing in the page colour, in the component's own shape: glass-like (translucent)
        # cards then look exactly as on the site instead of disappearing on a dark video background.
        inner = (f'<div style="display:block;background:{html_lib.escape(item["inheritedBackground"])};'
                 f'border-radius:{html_lib.escape(item.get("radius") or "0")}">{inner}</div>')
    # Cards keep their width (their layout depends on it); single-line parts get room to stay on one line.
    size = (f"width:{item['width']}px" if item["kind"] in ("card", "heading", "description")
            else f"width:max-content;min-width:{item['width']}px;white-space:nowrap")
    # <html>/<body> keep all their attributes (theme, fonts, direction); the page itself stays transparent.
    return (f'<!doctype html><html {bundle["htmlAttrs"]}>'
            f'<head><meta charset="utf-8"><base href="{html_lib.escape(bundle["base"])}">{bundle["styles"]}'
            "<style>*,*::before,*::after{animation:none!important;transition:none!important}"
            "html,body{background:transparent!important;margin:0!important;min-height:0!important}</style></head>"
            f'<body {bundle["bodyAttrs"]}>'
            f'<div id="mpt-root" style="display:inline-block;padding:24px;{size};box-sizing:content-box;'
            f'background:{background};border-radius:{radius}">{inner}</div></body></html>')


def render_components(browser, page, url: str, shots_dir: str, offline: bool, warnings: list) -> list[dict]:
    """Draw every real component as its own transparent PNG layer (no JavaScript, no page animation)."""
    from playwright.sync_api import Error as PlaywrightError

    try:
        bundle = page.evaluate(_COMPONENTS_JS, page.url)
    except PlaywrightError as exc:
        warnings.append(f"components could not be read: {str(exc).splitlines()[0][:160]}")
        return []
    layers_dir = os.path.join(shots_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    context = browser.new_context(viewport=DESKTOP, java_script_enabled=False, device_scale_factor=2,
                                  user_agent=USER_AGENT, accept_downloads=False)
    _prepare_context(context, url, "", offline)
    layers = []
    try:
        stage = context.new_page()
        for item in bundle["items"]:
            path = os.path.join(layers_dir, f"{item['id']}.png")
            try:
                stage.set_content(_component_page(bundle, item), wait_until="load", timeout=15000)
                stage.locator("#mpt-root").screenshot(path=path, omit_background=True, animations="disabled",
                                                     timeout=RETRY_TIMEOUT_MS)
            except PlaywrightError as exc:
                warnings.append(f"layer {item['id']}: {str(exc).splitlines()[0][:160]}")
                continue
            layers.append({"id": item["id"], "kind": item["kind"], "path": f"screenshots/layers/{item['id']}.png",
                           "text": item["text"], "layer": True, "width": item["width"], "height": item["height"]})
    finally:
        context.close()
    logger.info(f"rendered {len(layers)} component layers: {[layer['id'] for layer in layers]}")
    return layers


def _diagnose(page) -> dict:
    try:
        return page.evaluate(_DIAGNOSTICS_JS)
    except Exception as exc:
        return {"error": str(exc).splitlines()[0][:200]}


def _calm(page) -> None:
    try:
        page.add_style_tag(content=_CALM_CSS)
        page.evaluate(_CALM_JS)
        page.wait_for_timeout(400)  # a short moment for the page to settle
    except Exception as exc:
        logger.warning(f"could not calm the page before screenshots: {exc}")


def safe_screenshot(page, path: str, full_page: bool = False, clip: dict | None = None,
                    warnings: list | None = None) -> str:
    """Screenshot that never raises. Returns "full", "viewport" (fallback) or "" (no image).

    1st try: as asked, animations disabled, 20 s. 2nd try: the plainest shot of the visible
    viewport, 10 s. A page whose screenshot keeps timing out must not stop the analysis.
    """
    from playwright.sync_api import Error as PlaywrightError

    options = {"animations": "disabled", "caret": "hide", "timeout": SCREENSHOT_TIMEOUT_MS}
    if full_page:
        options["full_page"] = True
        if clip:
            options["clip"] = clip
    try:
        page.screenshot(path=path, **options)
        return "full" if full_page else "viewport"
    except PlaywrightError as first:
        logger.warning(f"screenshot {os.path.basename(path)} failed ({str(first).splitlines()[0]}); "
                       "retrying with the visible viewport only")
    try:
        page.screenshot(path=path, animations="disabled", timeout=RETRY_TIMEOUT_MS)
        if warnings is not None and full_page:
            warnings.append(f"{os.path.basename(path)}: only the visible part could be captured")
        return "viewport"
    except PlaywrightError as second:
        message = f"{os.path.basename(path)}: screenshot not possible ({str(second).splitlines()[0][:160]})"
        logger.warning(message + "; continuing without it")
        if warnings is not None:
            warnings.append(message)
        return ""


def _scroll_through(page, steps: int = 8) -> None:
    """Scroll down and back so lazy-loaded sections render."""
    height = page.evaluate("document.documentElement.scrollHeight")
    for i in range(1, steps + 1):
        page.evaluate(f"window.scrollTo(0, {int(min(height, MAX_PAGE_HEIGHT) * i / steps)})")
        page.wait_for_timeout(150)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)


def _cap_height(path: str) -> int:
    from PIL import Image

    with Image.open(path) as image:
        if image.height <= MAX_PAGE_HEIGHT:
            return image.height
        cropped = image.crop((0, 0, image.width, MAX_PAGE_HEIGHT))
    cropped.save(path)
    return MAX_PAGE_HEIGHT


def _crop(page_png: str, element_box: dict, out: str, pad: int = 0) -> bool:
    """Cut an element (page coordinates) out of the full-page screenshot."""
    from PIL import Image

    with Image.open(page_png) as image:
        left = max(0, int(element_box["x"]) - pad)
        top = max(0, int(element_box["y"]) - pad)
        right = min(image.width, int(element_box["x"] + element_box["w"]) + pad)
        bottom = min(image.height, int(element_box["y"] + element_box["h"]) + pad)
        if right - left < 20 or bottom - top < 12:
            return False
        image.crop((left, top, right, bottom)).save(out)
    return True


def _services_from_homepage(data: dict, page_url: str) -> list[dict]:
    """Main services on a homepage: links from service cards, then navigation links."""
    services, seen = [], set()
    skip = re.compile(r"login|sign ?in|sign ?up|register|contact|about|blog|privacy|terms|pricing|docs|faq|help|"
                      r"support|careers|home|upgrade|تسجيل|دخول|اتصل|من نحن|سياسة|الأسعار|الرئيسية|مساعدة",
                      re.IGNORECASE)
    for link in sorted(data["links"], key=lambda item: (not item["inCard"], not item["inNav"])):
        url = link["url"].rstrip("/")
        if url in seen or url == page_url.rstrip("/") or skip.search(link["text"] + " " + url):
            continue
        seen.add(url)
        services.append({"name": link["text"], "url": link["url"]})
        if len(services) >= 12:
            break
    return services


def build_facts(data: dict) -> tuple[list[dict], dict]:
    """Number every text so the ad can cite it; returns (texts, lookup by text)."""
    texts, index = [], {}

    def add(text: str, kind: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if text in index:
            return index[text]
        fact_id = f"t{len(texts) + 1}"
        texts.append({"id": fact_id, "kind": kind, "text": text})
        index[text] = fact_id
        return fact_id

    add(data.get("title", ""), "title")
    add(data.get("description", ""), "description")
    for heading in data["headings"]:
        add(heading["text"], f"h{heading['level']}")
    for card in data["cards"]:
        add(card["heading"], "card_title")
        add(card["text"], "card")
    for item in data["texts"][:MAX_TEXTS]:
        add(item["text"], "list_item" if item["inList"] else "paragraph")
    for cta in data["ctas"]:
        add(cta["text"], "cta")
    return texts, index


# Keys sites commonly use to remember the chosen language (cookie / localStorage).
LOCALE_KEYS = ("qaivo_locale", "NEXT_LOCALE", "locale", "lang", "language", "i18nextLng")

_SET_LOCALE_JS = """
(locale) => { for (const key of %s) { try { localStorage.setItem(key, locale); } catch (e) {} } }
""" % json.dumps(list(LOCALE_KEYS))


def _prepare_context(context, url: str, locale: str, offline: bool) -> None:
    """Ask the site for ``locale`` (cookies + localStorage) and, offline, block every other host."""
    if locale:
        host = urllib.parse.urlparse(url).hostname or ""
        context.add_cookies([{"name": key, "value": locale, "domain": host, "path": "/"} for key in LOCALE_KEYS])
        context.add_init_script(f"({_SET_LOCALE_JS})({json.dumps(locale)})")
    if offline:
        origin = urllib.parse.urlparse(url).netloc

        def only_local(route):
            if urllib.parse.urlparse(route.request.url).netloc == origin:
                route.continue_()
            else:
                route.abort()

        context.route("**/*", only_local)


def _switch_language(page, locale: str) -> None:
    """If the page is still in another language, use its own language menu (a <select>)."""
    if not locale or (page.evaluate("document.documentElement.lang") or "").lower().startswith(locale):
        return
    for select in page.query_selector_all("select"):
        values = page.evaluate("s => [...s.options].map(o => o.value)", select)
        if locale in values:
            try:
                select.select_option(locale)
                page.wait_for_timeout(1500)
            except Exception as exc:
                logger.warning(f"language switch failed: {exc}")
            return


def read_website(url: str, out_dir: str | None = None, portrait: bool = True, timeout_ms: int = 45000,
                 check_robots: bool = True, locale: str = "", offline: bool = False) -> dict:
    """Open the page, extract grounded facts and real screenshots, save ``website.json``.

    ``locale``: "ar"/"en"... ask the site for that language. ``offline``: block requests to any
    other host (used for a local copy of the site).
    """
    url = normalize_url(url)
    if check_robots and not robots_allowed(url):
        raise WebsiteError("This website's robots.txt does not allow automated reading of this page.")
    out_dir = out_dir or project_dir_for(url)
    shots_dir = os.path.join(out_dir, "screenshots")
    os.makedirs(shots_dir, exist_ok=True)
    browsers_path()
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    started = time.time()
    screenshots: list[dict] = []
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            raise WebsiteError("The browser for reading websites is not installed. Run install.bat again. "
                               f"({str(exc).splitlines()[0]})") from exc
        try:
            context = browser.new_context(viewport=DESKTOP, user_agent=USER_AGENT, accept_downloads=False,
                                          service_workers="block")
            _prepare_context(context, url, locale, offline)
            page = context.new_page()
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except PlaywrightError:
                    pass  # busy pages (chat widgets, analytics) never go idle
            except PlaywrightError as exc:
                raise WebsiteError(f"Could not open the page: {str(exc).splitlines()[0]}") from exc
            if response is not None and response.status >= 400:
                raise WebsiteError(f"The page answered with HTTP {response.status}.")
            _switch_language(page, locale)
            _scroll_through(page)
            data = page.evaluate(_EXTRACT_JS)
            if data["hasPassword"] and data["bodyText"] < 600:
                raise WebsiteError("This page is a login form. Give the public page of the service instead.")

            diagnostics = _diagnose(page)
            logger.info(f"page render diagnostics: {diagnostics}")
            _calm(page)
            shot_warnings: list[str] = []
            # Primary: the real components as separate layers. Page screenshots below are secondary.
            layers = render_components(browser, page, url, shots_dir, offline, shot_warnings)
            layer_ids = {layer["id"] for layer in layers}
            hero_path = os.path.join(shots_dir, "hero.png")
            full_path = os.path.join(shots_dir, "page.png")
            full_height = min(int(data["pageHeight"] or DESKTOP["height"]), MAX_PAGE_HEIGHT)
            full_kind = safe_screenshot(page, full_path, full_page=True, warnings=shot_warnings,
                                        clip={"x": 0, "y": 0, "width": DESKTOP["width"], "height": full_height})
            hero_kind = safe_screenshot(page, hero_path, warnings=shot_warnings)
            if not hero_kind and full_kind:
                hero_kind = "viewport" if _crop(full_path, {"x": 0, "y": 0, **{"w": DESKTOP["width"],
                                                "h": DESKTOP["height"]}}, hero_path) else ""
            if hero_kind:
                screenshots.append({"id": "hero", "kind": "hero", "path": "screenshots/hero.png",
                                    "text": data["title"], **DESKTOP})
            if full_kind:
                full_height = _cap_height(full_path)
                screenshots.append({"id": "page", "kind": "page", "path": "screenshots/page.png",
                                    "text": data["title"], "width": DESKTOP["width"], "height": full_height})
            else:
                full_path = hero_path if hero_kind else ""
            # Parts are cut out of the full-page shot: no waiting for animated elements to settle.
            for index, card in enumerate(data["cards"][:MAX_CARDS]):
                name = f"card{index + 1}.png"
                if f"card{index + 1}" in layer_ids:
                    continue
                if full_path and _crop(full_path, card["box"], os.path.join(shots_dir, name)):
                    screenshots.append({"id": f"card{index + 1}", "kind": "card", "path": f"screenshots/{name}",
                                        "text": card["heading"], "detail": card["text"]})
            for index, cta in enumerate(data["ctas"][:MAX_CTAS]):
                name = f"cta{index + 1}.png"
                if f"cta{index + 1}" in layer_ids:
                    continue
                if full_path and _crop(full_path, cta["box"], os.path.join(shots_dir, name), pad=4):
                    screenshots.append({"id": f"cta{index + 1}", "kind": "cta", "path": f"screenshots/{name}",
                                        "text": cta["text"]})
            if "logo" not in layer_ids and full_path and data.get("logoBox") and _crop(full_path, data["logoBox"],
                                                            os.path.join(shots_dir, "logo.png"), pad=6):
                screenshots.append({"id": "logo", "kind": "logo", "path": "screenshots/logo.png",
                                    "text": (data["logo"] or {}).get("alt", "")})
            for layer in layers:  # the independent components come first
                if layer["kind"] == "card":
                    index = int(layer["id"][4:]) - 1
                    if index < len(data["cards"]):
                        layer = dict(layer, text=data["cards"][index]["heading"], detail=data["cards"][index]["text"])
                screenshots.append(layer)
            if portrait:
                mobile = browser.new_context(viewport=MOBILE, user_agent=USER_AGENT, accept_downloads=False,
                                             is_mobile=True, has_touch=True, service_workers="block")
                _prepare_context(mobile, url, locale, offline)
                mobile_page = mobile.new_page()
                try:
                    mobile_page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    try:
                        mobile_page.wait_for_load_state("networkidle", timeout=10000)
                    except PlaywrightError:
                        pass
                    _switch_language(mobile_page, locale)
                    _calm(mobile_page)
                    if safe_screenshot(mobile_page, os.path.join(shots_dir, "mobile.png"), warnings=shot_warnings):
                        screenshots.append({"id": "mobile", "kind": "mobile", "path": "screenshots/mobile.png",
                                            "text": data["title"], **MOBILE})
                except PlaywrightError as exc:
                    logger.warning(f"mobile screenshot failed: {exc}")
            final_url = page.url
        finally:
            browser.close()

    visible = next((layer["text"] for layer in layers if layer["kind"] == "description"), "")
    if visible:  # the intro under the heading is in the page's language; <meta> often is not
        data["description"] = visible
    texts, index = build_facts(data)
    homepage = is_homepage(final_url)
    h1 = next((h["text"] for h in data["headings"] if h["level"] == 1), "")
    service_name = h1 or data["title"]
    benefits = []
    for card in data["cards"][:MAX_CARDS]:
        benefits.append({"id": index.get(card["heading"], ""), "text": card["heading"],
                         "detail": card["text"], "screenshot": next(
                             (s["id"] for s in screenshots if s["kind"] == "card" and s["text"] == card["heading"]), "")})
    for item in data["texts"]:
        if item["inList"] and len(benefits) < 12 and len(item["text"]) <= 160:
            benefits.append({"id": index[item["text"]], "text": item["text"], "detail": "", "screenshot": ""})
    result = {
        "url": url,
        "final_url": final_url,
        "mode": "homepage" if homepage else "service",
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - started, 1),
        "lang": data["lang"],
        "dir": data["dir"],
        "brand": {"name": data["siteName"] or (data["title"].split("|")[-1].split("-")[-1].strip() if data["title"]
                                               else urllib.parse.urlparse(final_url).netloc),
                  "logo": "screenshots/logo.png" if any(s["id"] == "logo" for s in screenshots) else "",
                  "colors": [c for c in data["colors"] if c][:6]},
        "service": {"name": service_name, "name_id": index.get(service_name, ""),
                    "description": data["description"], "description_id": index.get(data["description"], "")},
        "benefits": benefits,
        "cta": [{"id": index.get(c["text"], ""), "text": c["text"], "href": c["href"],
                 "screenshot": f"cta{i + 1}" if any(s["id"] == f"cta{i + 1}" for s in screenshots) else ""}
                for i, c in enumerate(data["ctas"])],
        "services": _services_from_homepage(data, final_url) if homepage else [],
        "texts": texts,
        "assets": [{"type": "image", "src": img["src"], "alt": img["alt"]} for img in data["images"]],
        "screenshots": screenshots,
        "screenshot_warnings": shot_warnings,
        "render_diagnostics": diagnostics,
        "source_urls": [final_url],
    }
    with open(os.path.join(out_dir, "website.json"), "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    logger.info(f"website read: {final_url} ({len(texts)} texts, {len(screenshots)} screenshots, "
                f"{result['seconds']}s)")
    result["_dir"] = out_dir
    return result


def load_website(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "website.json"), encoding="utf-8") as fp:
        data = json.load(fp)
    data["_dir"] = out_dir
    return data
