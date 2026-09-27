import os
import subprocess
import tempfile
import unittest

import numpy as np
from moviepy import VideoFileClip
from PIL import Image, ImageDraw

from app.services.marketing import compositor as c
from app.utils import utils

FFMPEG = utils.get_ffmpeg_binary()
PRESENTER = (200, 40, 40)  # bright red "presenter" so we can find her in frames


def _presenter_on(bg, path, size=(480, 832)):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    draw.ellipse([180, 120, 300, 260], fill=PRESENTER)
    draw.rounded_rectangle([120, 270, 360, size[1]], 60, fill=PRESENTER)
    img.save(path)
    return path


class Assets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.page = os.path.join(self.tmp, "page.png")
        Image.new("RGB", (1440, 2400), (250, 250, 250)).save(self.page)
        self.card = os.path.join(self.tmp, "card.png")
        Image.new("RGB", (360, 200), (20, 20, 220)).save(self.card)
        self.green = _presenter_on((0, 255, 0), os.path.join(self.tmp, "green.png"))
        self.grey = _presenter_on((190, 190, 190), os.path.join(self.tmp, "grey.png"))


class TestPieces(Assets):
    def test_palette_picks_dark_and_accent(self):
        pal = c.palette(["#f8fafc", "#0f172a", "#10b981"])
        self.assertEqual(pal["dark"], (15, 23, 42))
        self.assertEqual(pal["accent"], (16, 185, 129))
        self.assertEqual(c.palette([])["dark"], (15, 23, 42))  # defaults
        self.assertLess(c._luma(c.palette(["#ffffff"])["dark"]), 90)

    def test_green_and_plain_backgrounds_are_keyed(self):
        for path in (self.green, self.grey):
            frame = np.asarray(Image.open(path).convert("RGB"))
            keyed = c.key_frame(frame, c.key_color(frame))
            alpha = np.asarray(keyed.getchannel("A"))
            self.assertEqual(alpha[5, 5], 0)  # background gone
            self.assertEqual(alpha[600, 240], 255)  # body kept
            self.assertEqual(tuple(np.asarray(keyed)[600, 240][:3]), PRESENTER)

    def test_green_spill_is_reduced(self):
        frame = np.full((50, 50, 3), (0, 255, 0), np.uint8)
        frame[20:30, 20:30] = (120, 200, 110)  # greenish skin edge
        keyed = np.asarray(c.key_frame(frame, np.array([0, 255, 0])))
        self.assertLessEqual(int(keyed[25, 25, 1]), 120 + 8)

    def test_turn_makes_a_perspective_panel(self):
        panel = Image.new("RGBA", (400, 300), (255, 255, 255, 255))
        turned = c.turn(panel, 0.2, "right")
        self.assertLess(turned.width, 400)
        alpha = np.asarray(turned.getchannel("A"))
        self.assertEqual(alpha[2, 2], 255)  # near edge full height
        self.assertEqual(alpha[2, turned.width - 3], 0)  # far edge shorter

    def test_presenter_source_from_video_keeps_one_crop(self):
        video = os.path.join(self.tmp, "talk.mp4")
        subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-loop", "1", "-i", self.green, "-t", "1", "-r", "25",
                        "-pix_fmt", "yuv420p", video], check=True)
        source = c.PresenterSource(video=video, height=300)
        self.assertGreaterEqual(len(source.frames), 24)
        self.assertEqual({f.size for f in source.frames}.__len__(), 1)
        self.assertEqual(source.frames[0].height, 300)
        self.assertIsNone(c.PresenterSource().frame(0))


def _red_columns(frame):
    red = (frame[..., 0] > 150) & (frame[..., 1] < 90) & (frame[..., 2] < 90)
    return np.where(red.any(axis=0))[0]


class TestScenes(Assets):
    def _render(self, spec, w=640, h=360, seconds=1.2):
        out = os.path.join(self.tmp, f"scene_{len(os.listdir(self.tmp))}.mp4")
        c.render_scene({"colors": ["#0f172a", "#10b981"], **spec}, out, w, h, seconds, preset="ultrafast")
        with VideoFileClip(out) as clip:
            self.assertEqual(clip.size, [w, h])
            self.assertAlmostEqual(clip.duration, seconds, delta=0.1)
            return clip.get_frame(seconds - 0.1)

    def test_world_scene_places_presenter_left_and_card_in_frame(self):
        frame = self._render({"style": "world", "screen": self.page, "cards": [self.card, self.card],
                              "presenter_image": self.green})
        columns = _red_columns(frame)
        self.assertTrue(len(columns) and columns.mean() < 640 * 0.45)
        blue = (frame[..., 2] > 150) & (frame[..., 0] < 80) & (frame[..., 1] < 80)
        self.assertGreater(blue.sum(), 500)  # the real cards are visible
        self.assertFalse(np.any(frame[..., 1] > 240) and (frame[..., 0] < 20).all())  # no green left

    def test_arabic_layout_puts_presenter_on_the_right(self):
        frame = self._render({"style": "world", "rtl": True, "screen": self.page, "presenter_image": self.green})
        self.assertGreater(_red_columns(frame).mean(), 640 * 0.55)

    def test_portrait_presenter_stands_on_the_bottom(self):
        frame = self._render({"style": "world", "screen": self.page, "cards": [self.card],
                              "presenter_image": self.green}, w=360, h=640)
        red_rows = np.where(((frame[..., 0] > 150) & (frame[..., 1] < 90)).any(axis=1))[0]
        self.assertGreater(red_rows.max(), 640 * 0.95)
        self.assertGreater(red_rows.min(), 640 * 0.35)

    def test_screen_scene_and_cta(self):
        frame = self._render({"style": "screen", "screen": self.page})
        self.assertGreater(frame[180, 320].sum(), 600)  # the white page in the middle
        frame = self._render({"style": "cta", "cta": self.card, "logo": self.card, "presenter_image": self.grey})
        self.assertTrue(len(_red_columns(frame)))

    def test_missing_assets_still_render(self):
        self._render({"style": "world", "screen": "/nope.png", "cards": ["/nope.png"]})


if __name__ == "__main__":
    unittest.main()
