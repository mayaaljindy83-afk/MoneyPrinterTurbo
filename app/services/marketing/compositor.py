"""Pseudo-3D "website world" scenes: real screenshots + presenter + camera, no 3D engine.

A scene is a stack of layers, each at a depth:

    background (brand gradient, light, floor grid; or an AI environment image)   depth 0.15
    big screen: the real page in a browser frame, turned in perspective + shadow depth 0.45
    presenter: keyed (green screen or plain studio background) video or photo    depth 0.8
    floating cards: real cards / buttons cut from the page, shadow, bobbing      depth 1.0

The camera slowly pushes in and pans; nearer layers move more (parallax), the
background is blurred (depth of field) and a vignette finishes the look.
Every word on screen comes from real screenshots: nothing is redrawn.

Frames are drawn with Pillow/numpy and piped straight into FFmpeg.
"""

from __future__ import annotations

import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.utils import utils

FPS = 25


# --------------------------------------------------------------------------- colours
def hex_rgb(value: str, default=(15, 23, 42)) -> tuple[int, int, int]:
    value = (value or "").strip().lstrip("#")
    if len(value) != 6:
        return default
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def _luma(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def palette(colors: list[str]) -> dict:
    """dark (base), accent (glow), light, from the site colours with sensible defaults."""
    rgbs = [hex_rgb(c, None) for c in colors or []]
    rgbs = [c for c in rgbs if c]
    dark = min(rgbs, key=_luma) if rgbs else (15, 23, 42)
    if _luma(dark) > 90:
        dark = tuple(int(v * 0.25) for v in dark)
    saturated = [c for c in rgbs if max(c) - min(c) > 60]
    accent = max(saturated, key=lambda c: max(c) - min(c)) if saturated else (16, 185, 129)
    return {"dark": dark, "accent": accent, "light": (248, 250, 252)}


# --------------------------------------------------------------------------- layer building
def background(width: int, height: int, pal: dict, image: str | None = None, seed: int = 0) -> Image.Image:
    """A brand-coloured space: gradient, soft light blobs and a perspective floor grid."""
    if image and os.path.isfile(image):
        base = Image.open(image).convert("RGB")
        scale = max(width / base.width, height / base.height)
        base = base.resize((math.ceil(base.width * scale), math.ceil(base.height * scale)), Image.LANCZOS)
        left, top = (base.width - width) // 2, (base.height - height) // 2
        return base.crop((left, top, left + width, top + height)).convert("RGBA")
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    dark = np.array(pal["dark"], float)
    accent = np.array(pal["accent"], float)
    canvas = dark * (0.75 + 0.35 * y)[..., None] * np.ones((1, width, 1))
    for _ in range(3):  # soft coloured lights
        cx, cy, radius = rng.uniform(0.1, 0.9), rng.uniform(0.05, 0.6), rng.uniform(0.25, 0.5)
        glow = np.exp(-(((x - cx) * width / height) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))
        canvas += glow[..., None] * accent * rng.uniform(0.18, 0.32)
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).convert("RGBA")
    grid = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid)
    horizon = int(height * 0.62)
    color = tuple(int(v) for v in accent) + (40,)
    for i in range(-14, 15):  # lines converging to the horizon
        draw.line([(width / 2 + i * width * 0.02, horizon), (width / 2 + i * width * 0.16, height)], fill=color, width=1)
    for k in range(1, 9):
        yy = horizon + (height - horizon) * (k / 8) ** 1.8
        draw.line([(0, yy), (width, yy)], fill=color, width=1)
    img.alpha_composite(grid)
    dust = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ddraw = ImageDraw.Draw(dust)
    for _ in range(60):  # floating particles
        px, py, r = rng.uniform(0, width), rng.uniform(0, height * 0.7), rng.uniform(1, 3.5)
        ddraw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255, int(rng.uniform(20, 70))))
    img.alpha_composite(dust.filter(ImageFilter.GaussianBlur(1)))
    return img


def rounded(img: Image.Image, radius: int) -> Image.Image:
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=radius, fill=255)
    img.putalpha(Image.fromarray(np.minimum(np.asarray(img.getchannel("A")), np.asarray(mask))))
    return img


def browser_frame(screenshot: Image.Image, width: int, max_height: int, pal: dict) -> Image.Image:
    """The page inside a simple browser window (bar + three dots, no invented text)."""
    shot = screenshot.convert("RGB")
    scale = width / shot.width
    shot = shot.resize((width, max(1, int(shot.height * scale))), Image.LANCZOS)
    bar = max(18, width // 40)
    height = min(max_height, shot.height + bar)
    frame = Image.new("RGBA", (width, height), (30, 36, 48, 255))
    draw = ImageDraw.Draw(frame)
    for i, dot in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        cx = bar * (0.8 + i * 0.7)
        draw.ellipse([cx - bar * 0.2, bar * 0.3, cx + bar * 0.2, bar * 0.7], fill=dot)
    draw.rounded_rectangle([bar * 3, bar * 0.22, width - bar, bar * 0.78], radius=int(bar * 0.28),
                           fill=(52, 60, 74))
    frame.paste(shot.crop((0, 0, width, height - bar)), (0, bar))
    return rounded(frame, max(6, width // 60))


def _perspective_coeffs(dst, src):
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    return np.linalg.solve(np.array(rows, float), np.array(rhs, float)).tolist()


def turn(img: Image.Image, amount: float, facing: str = "right") -> Image.Image:
    """Turn a flat panel around its vertical axis (pseudo-3D). ``amount`` 0..0.3."""
    w, h = img.size
    new_w = int(w * (1 - amount * 0.35))
    squeeze = h * amount * 0.5
    if facing == "right":  # left edge near, right edge far
        dst = [(0, 0), (new_w, squeeze), (new_w, h - squeeze), (0, h)]
    else:
        dst = [(0, squeeze), (new_w, 0), (new_w, h), (0, h - squeeze)]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    return img.transform((new_w, h), Image.PERSPECTIVE, _perspective_coeffs(dst, src), Image.BICUBIC)


def with_shadow(img: Image.Image, blur: int = 18, offset=(0, 14), opacity: float = 0.55,
                glow: tuple | None = None) -> tuple[Image.Image, tuple[int, int]]:
    """Returns (image with shadow/glow around it, offset of the original inside it)."""
    pad = blur * 3
    canvas = Image.new("RGBA", (img.width + pad * 2, img.height + pad * 2), (0, 0, 0, 0))
    alpha = img.getchannel("A")
    if glow:
        halo = Image.new("RGBA", img.size, glow + (0,))
        halo.putalpha(alpha.point(lambda a: int(a * 0.55)))
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(halo, (pad, pad), halo)
        canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur * 1.4)))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda a: int(a * opacity)))
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(shadow, (pad + offset[0], pad + offset[1]), shadow)
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))
    canvas.alpha_composite(img, (pad, pad))
    return canvas, (pad, pad)


def card(img_path: str, max_w: int, max_h: int) -> Image.Image:
    img = Image.open(img_path).convert("RGBA")
    scale = min(max_w / img.width, max_h / img.height, 2.0)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    return rounded(img, max(6, img.width // 25))


# --------------------------------------------------------------------------- presenter keying
def key_color(frame: np.ndarray) -> np.ndarray:
    """Background colour of a presenter shot: median of the top and side borders."""
    h, w, _ = frame.shape
    border = np.concatenate([frame[: max(2, h // 30)].reshape(-1, 3),
                             frame[:, : max(2, w // 40)].reshape(-1, 3),
                             frame[:, -max(2, w // 40):].reshape(-1, 3)])
    return np.median(border, axis=0)


def key_frame(frame: np.ndarray, color: np.ndarray, low: float = 40, high: float = 95) -> Image.Image:
    """RGB frame -> RGBA with the flat background removed and colour spill reduced."""
    rgb = frame.astype(np.float32)
    distance = np.sqrt(((rgb - color) ** 2).sum(axis=2))
    alpha = np.clip((distance - low) / (high - low), 0, 1)
    if color[1] > color[0] + 40 and color[1] > color[2] + 40:  # green screen: remove green spill
        limit = np.maximum(rgb[..., 0], rgb[..., 2])
        rgb[..., 1] = np.minimum(rgb[..., 1], limit + 8)
    out = np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8)
    img = Image.fromarray(out)
    # Soften the edge by a pixel so it does not look cut out.
    img.putalpha(img.getchannel("A").filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8)))
    return img


class PresenterSource:
    """Keyed presenter frames from a video (lip-synced shot) or a still photo."""

    def __init__(self, video: str = "", image: str = "", height: int = 720, fps: int = FPS):
        self.height = height
        self.frames: list[Image.Image] = []
        self.index = 0
        if video and os.path.isfile(video):
            self._load_video(video, fps)
        elif image and os.path.isfile(image):
            still = np.asarray(Image.open(image).convert("RGB"))
            self.frames = [self._fit(key_frame(still, key_color(still)))]

    def _fit(self, img: Image.Image) -> Image.Image:
        bbox = img.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
        if bbox:
            img = img.crop((bbox[0], bbox[1], bbox[2], img.height))
        scale = self.height / img.height
        return img.resize((max(1, int(img.width * scale)), self.height), Image.LANCZOS)

    def _load_video(self, video: str, fps: int) -> None:
        probe = subprocess.run([utils.get_ffmpeg_binary(), "-i", video], capture_output=True, text=True).stderr
        import re

        size = re.search(r", (\d{2,5})x(\d{2,5})", probe)
        if not size:
            raise RuntimeError(f"cannot read presenter video {video}")
        w, h = int(size.group(1)), int(size.group(2))
        raw = subprocess.run([utils.get_ffmpeg_binary(), "-loglevel", "error", "-i", video, "-vf", f"fps={fps}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        count = len(raw) // (w * h * 3)
        frames = np.frombuffer(raw[: count * w * h * 3], np.uint8).reshape(count, h, w, 3)
        color = key_color(frames[0])
        crop = None
        for frame in frames:
            keyed = key_frame(frame, color)
            if crop is None:  # same crop for every frame so the presenter does not jump
                crop = keyed.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox() or (0, 0, w, h)
                pad = int(w * 0.05)
                crop = (max(0, crop[0] - pad), max(0, crop[1] - pad), min(w, crop[2] + pad), h)
            img = keyed.crop(crop)
            scale = self.height / img.height
            self.frames.append(img.resize((max(1, int(img.width * scale)), self.height), Image.BILINEAR))

    def frame(self, i: int) -> Image.Image | None:
        if not self.frames:
            return None
        return self.frames[min(i, len(self.frames) - 1)]


# --------------------------------------------------------------------------- scene
def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _affine(layer: Image.Image, size, pos, scale: float, center) -> tuple[Image.Image, tuple[int, int]]:
    """Place ``layer`` (top-left ``pos`` in scene space) under a camera zoom around ``center``."""
    w = max(1, int(layer.width * scale))
    h = max(1, int(layer.height * scale))
    x = center[0] + (pos[0] - center[0]) * scale
    y = center[1] + (pos[1] - center[1]) * scale
    return layer.resize((w, h), Image.BILINEAR), (int(round(x)), int(round(y)))


def _paste(canvas: Image.Image, layer: Image.Image, xy, opacity: float = 1.0) -> None:
    if opacity < 1:
        layer = layer.copy()
        layer.putalpha(layer.getchannel("A").point(lambda a: int(a * opacity)))
    canvas.alpha_composite(layer, dest=(max(0, xy[0]), max(0, xy[1])),
                           source=(max(0, -xy[0]), max(0, -xy[1])))


def layout(style: str, width: int, height: int, rtl: bool) -> dict:
    """Where things go, as fractions of the scene.

    screen: (x, y, width, max height); presenter: (centre x, height), always standing on the
    bottom edge; cards: top-left corners; card: card width.
    """
    if height > width:  # 9:16: page on top, presenter below, cards beside her
        return {"screen": (0.06, 0.07, 0.88, 0.40), "presenter": (0.30 if not rtl else 0.70, 0.56),
                "cards": [(0.52 if not rtl else 0.04, 0.40), (0.56 if not rtl else 0.02, 0.60)],
                "card": 0.42, "facing": "right" if not rtl else "left"}
    presenter_x = 0.76 if rtl else 0.24
    screen_x = 0.06 if rtl else 0.40
    facing = "left" if rtl else "right"
    if style == "point":
        return {"screen": (screen_x + (0.04 if rtl else 0.0), 0.14, 0.50, 0.62), "presenter": (presenter_x, 0.86),
                "cards": [(0.50 if rtl else 0.33, 0.50)], "card": 0.30, "facing": facing}
    return {"screen": (screen_x, 0.10, 0.54, 0.62), "presenter": (presenter_x, 0.88),
            "cards": [(0.50 if rtl else 0.30, 0.56), (0.06 if rtl else 0.74, 0.62)], "card": 0.22,
            "facing": facing}


def _clamp(x: float, y: float, img: Image.Image, sw: int, sh: int, pad: int = 0, over: float = 1.12) -> tuple[int, int]:
    """Keep a layer inside the visible frame (the scene is ``over`` times the frame), with a margin
    for the camera push-in."""
    fw, fh = sw / over, sh / over
    left, top = (sw - fw) / 2 + fw * 0.03, (sh - fh) / 2 + fh * 0.03
    right, bottom = (sw + fw) / 2 - fw * 0.03, (sh + fh) / 2 - fh * 0.03
    return (int(min(max(x, left - pad), right - img.width + pad)),
            int(min(max(y, top - pad), bottom - img.height + pad)))


def render_scene(spec: dict, output: str, width: int, height: int, seconds: float, fps: int = FPS,
                 preset: str = "veryfast") -> str:
    """Render one composited scene.

    spec keys: style ("world" | "point" | "screen" | "cta"), colors, background (image, optional),
    screen (screenshot), cards [paths], presenter_video / presenter_image, logo, cta (button
    screenshot), rtl (presenter on the right), seed.
    """
    style = spec.get("style", "world")
    pal = palette(spec.get("colors") or [])
    frames = max(1, int(round(seconds * fps)))
    over = 1.12  # the scene is drawn bigger than the frame so the camera can move
    sw, sh = int(width * over), int(height * over)
    offset = ((sw - width) // 2, (sh - height) // 2)
    rtl = bool(spec.get("rtl"))
    plan = layout(style, width, height, rtl)

    bg = background(sw, sh, pal, spec.get("background"), spec.get("seed", 0))
    bg = bg.filter(ImageFilter.GaussianBlur(3 if spec.get("background") else 1.2))
    layers = []  # (image, (x, y) in scene space, depth, kind, index)

    screen_path = spec.get("screen")
    if screen_path and os.path.isfile(screen_path):
        shot = Image.open(screen_path)
        if style == "screen":
            panel_w = int(sw * (0.60 if height > width else 0.74))
            panel = browser_frame(shot, panel_w, int(sh * 2.2), pal)
            layers.append((panel, ((sw - panel_w) // 2, int(sh * 0.12)), 0.45, "scroll", 0))
        else:
            fx, fy, fw, fh = plan["screen"]
            panel = browser_frame(shot, int(sw * fw), int(sh * fh), pal)
            panel = turn(panel, 0.16, plan["facing"])
            panel, pad = with_shadow(panel, blur=max(10, sw // 90), glow=pal["accent"])
            layers.append((panel, (int(sw * fx) - pad[0], int(sh * fy) - pad[1]), 0.45, "screen", 0))

    card_paths = [p for p in spec.get("cards") or [] if p and os.path.isfile(p)]
    if style == "cta" and spec.get("cta") and os.path.isfile(spec["cta"]):
        button = card(spec["cta"], int(sw * 0.42), int(sh * 0.16))
        button, pad = with_shadow(button, blur=14, glow=pal["accent"])
        cx = 0.30 if rtl else 0.62
        layers.append((button, (int(sw * cx - button.width / 2), int(sh * 0.58)), 1.0, "pulse", 0))
    for index, path in enumerate(card_paths[: len(plan["cards"])] if style != "cta" else []):
        fx, fy = plan["cards"][index]
        big = style == "point"
        item = card(path, int(sw * plan["card"]), int(sh * (0.34 if big else 0.26)))
        item = turn(item, 0.06, "left" if (index % 2) else "right")
        item, pad = with_shadow(item, blur=12, glow=pal["accent"] if big else None)
        layers.append((item, _clamp(sw * fx - pad[0], sh * fy - pad[1], item, sw, sh, pad[0]), 1.0, "card", index))
    if style == "cta" and spec.get("logo") and os.path.isfile(spec["logo"]):
        logo = card(spec["logo"], int(sw * 0.30), int(sh * 0.14))
        logo, pad = with_shadow(logo, blur=10, opacity=0.35)
        cx = 0.30 if rtl else 0.62
        layers.append((logo, (int(sw * cx - logo.width / 2), int(sh * 0.30)), 0.7, "logo", 0))

    presenter = None
    if style != "screen":
        p_height = int(sh * plan["presenter"][1])
        presenter = PresenterSource(spec.get("presenter_video", ""), spec.get("presenter_image", ""), p_height, fps)
        if not presenter.frames:
            presenter = None
    vignette = _vignette(width, height)

    command = [utils.get_ffmpeg_binary(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset,
               "-crf", "18", "-pix_fmt", "yuv420p", output]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(frames):
            t = i / max(1, frames - 1)
            e = _ease(t)
            zoom = 1.0 + 0.07 * e
            pan = (0.5 - e) * width * 0.035 * (-1 if rtl else 1)
            center = (sw / 2, sh / 2)
            canvas = Image.new("RGBA", (sw, sh), (0, 0, 0, 255))
            bg_img, bg_xy = _affine(bg, (sw, sh), (pan * 0.15, 0), 1 + (zoom - 1) * 0.3, center)
            _paste(canvas, bg_img, bg_xy)
            presenter_done = presenter is None
            for image, (x, y), depth, kind, index in sorted(layers, key=lambda layer: layer[2]):
                if not presenter_done and depth > 0.8:
                    _paste_presenter(canvas, presenter, i, plan, sw, sh, zoom, pan, center)
                    presenter_done = True
                scale = 1 + (zoom - 1) * depth
                dx = pan * depth
                opacity = 1.0
                if kind == "card":
                    start = 0.25 + 0.3 * index
                    appear = min(1.0, max(0.0, (i / fps - start) / 0.5))
                    opacity = appear
                    y += int((1 - _ease(appear)) * 40 + math.sin(i / fps * 1.6 + index) * 6)
                elif kind == "pulse":
                    scale *= 1 + 0.03 * math.sin(i / fps * 4)
                elif kind == "scroll":
                    travel = max(0, image.height - sh * 0.8)
                    y -= int(travel * e * 0.6)
                if opacity <= 0:
                    continue
                img, xy = _affine(image, (sw, sh), (x + dx, y), scale, center)
                _paste(canvas, img, xy, opacity)
            if not presenter_done:
                _paste_presenter(canvas, presenter, i, plan, sw, sh, zoom, pan, center)
            frame = canvas.crop((offset[0], offset[1], offset[0] + width, offset[1] + height))
            frame.alpha_composite(vignette)
            process.stdin.write(frame.convert("RGB").tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {process.stderr.read().decode(errors='replace')[-800:]}")
    finally:
        if process.poll() is None:
            process.kill()
    return output


def _paste_presenter(canvas, presenter: PresenterSource, i, plan, sw, sh, zoom, pan, center) -> None:
    person = presenter.frame(i)
    fx, _ = plan["presenter"]
    depth = 0.8
    breathe = 1 + 0.006 * math.sin(i / FPS * 1.3)  # a still photo still feels alive
    scale = (1 + (zoom - 1) * depth) * breathe
    x = sw * fx - person.width / 2 + pan * depth
    y = sh - person.height * 0.97  # standing on the bottom edge (knees-up framing)
    shadow = Image.new("RGBA", (int(person.width * 0.9), max(8, person.width // 8)), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([0, 0, shadow.width - 1, shadow.height - 1], fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow.height / 3))
    img, xy = _affine(shadow, None, (x + person.width * 0.05, y + person.height - shadow.height / 2), scale, center)
    _paste(canvas, img, xy)
    img, xy = _affine(person, None, (x, y), scale, center)
    _paste(canvas, img, xy)


def _vignette(width: int, height: int) -> Image.Image:
    y = np.linspace(-1, 1, height)[:, None]
    x = np.linspace(-1, 1, width)[None, :]
    strength = np.clip((x ** 2 + y ** 2) ** 0.5 - 0.55, 0, 1) * 150
    return Image.fromarray(np.dstack([np.zeros((height, width, 3)), strength]).astype(np.uint8))
