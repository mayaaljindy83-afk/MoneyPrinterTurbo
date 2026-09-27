import os
import subprocess
import tempfile
import unittest
from unittest import mock

from moviepy import VideoFileClip
from PIL import Image

from app.config import config
from app.models.schema import VideoConcatMode, VideoParams
from app.services import branding, task, video
from app.utils import utils

FFMPEG = utils.get_ffmpeg_binary()


def _logo(path, size=(400, 100)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.paste((99, 102, 241, 255), (20, 20, size[0] - 20, size[1] - 20))
    image.save(path)


class BrandingFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.brand = os.path.join(self.tmp, "branding")
        os.makedirs(os.path.join(self.brand, "screenshots"))
        self.patch = mock.patch.dict(config.app, {"branding_dir": self.brand})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()


class TestFindParts(BrandingFolder):
    def test_portrait_variant_is_preferred_for_vertical_videos(self):
        for name in ("intro.png", "intro_portrait.png", "outro.png"):
            open(os.path.join(self.brand, name), "wb").close()
        self.assertTrue(branding.find_part("intro", portrait=True).endswith("intro_portrait.png"))
        self.assertTrue(branding.find_part("intro").endswith("intro.png"))
        # No portrait outro: fall back to the landscape one.
        self.assertTrue(branding.find_part("outro", portrait=True).endswith("outro.png"))

    def test_screenshots_follow_orientation(self):
        for name in ("desktop_home.png", "mobile_home.png", "portrait_x.jpg", "notes.txt"):
            open(os.path.join(self.brand, "screenshots", name), "wb").close()
        landscape = [os.path.basename(f) for f in branding.list_screenshots(portrait=False)]
        portrait = [os.path.basename(f) for f in branding.list_screenshots(portrait=True)]
        self.assertEqual(landscape, ["desktop_home.png"])
        self.assertEqual(portrait, ["mobile_home.png", "portrait_x.jpg"])


class TestWatermark(BrandingFolder):
    def test_watermark_image_has_plate_and_width(self):
        logo = os.path.join(self.brand, "logo.png")
        _logo(logo)
        out = branding.watermark_image(logo, 200, os.path.join(self.tmp, "wm.png"))
        image = Image.open(out)
        self.assertEqual(image.mode, "RGBA")
        self.assertGreater(image.width, 200)
        # Corner is transparent (rounded plate), centre is the logo colour.
        self.assertEqual(image.getpixel((0, 0))[3], 0)
        centre = image.getpixel((image.width // 2, image.height // 2))
        self.assertEqual(centre[:3], (99, 102, 241))

    def test_no_logo_returns_none(self):
        self.assertIsNone(branding.watermark_clip(1280, 720, 2, self.tmp))

    def test_generate_video_draws_logo_top_right(self):
        _logo(os.path.join(self.brand, "logo.png"))
        bg = os.path.join(self.tmp, "bg.mp4")
        audio = os.path.join(self.tmp, "a.mp3")
        out = os.path.join(self.tmp, "final.mp4")
        subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=black:s=1280x720:d=2:r=30", bg], check=True)
        subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "anullsrc=r=44100:cl=mono", "-t", "2", audio], check=True)
        with mock.patch.dict(config.app, {"video_resolution": "720p"}):
            params = VideoParams(video_subject="x", video_aspect="16:9", subtitle_enabled=False,
                                 bgm_type="", add_logo_watermark=True, n_threads=1)
            video.generate_video(bg, audio, "", out, params)
        with VideoFileClip(out) as clip:
            frame = clip.get_frame(1.0)
        top_right = frame[:150, 1000:].reshape(-1, 3)
        bottom_left = frame[570:, :280].reshape(-1, 3)
        self.assertGreater((top_right.sum(axis=1) > 400).sum(), 500)  # white plate
        self.assertEqual((bottom_left.sum(axis=1) > 60).sum(), 0)


class TestScreenshots(BrandingFolder):
    def test_screenshot_clip_matches_canvas(self):
        shot = os.path.join(self.brand, "screenshots", "mobile_home.png")
        Image.new("RGB", (430, 932), (255, 255, 255)).save(shot)
        clips = branding.screenshot_clips(720, 1280, 3, self.tmp)
        self.assertEqual(len(clips), 1)
        with VideoFileClip(clips[0]) as clip:
            self.assertEqual(clip.size, [720, 1280])
            self.assertAlmostEqual(clip.duration, 4.0, delta=0.1)
            frame = clip.get_frame(0.1)
        self.assertGreater(frame[640, 360].sum(), 700)  # the white page in the middle
        self.assertLess(frame[5, 5].sum(), 450)  # darkened blurred border

    def test_pipeline_mix_keeps_order_and_limits_count(self):
        for i in range(6):
            Image.new("RGB", (1440, 900), (200, 200, 200)).save(
                os.path.join(self.brand, "screenshots", f"desktop_{i}.png")
            )
        params = VideoParams(video_subject="x", video_aspect="16:9", add_site_screenshots=True)
        with mock.patch.object(task.utils, "task_dir", return_value=self.tmp), mock.patch.dict(
            config.app, {"video_resolution": "720p"}
        ):
            mixed = task._mix_in_screenshots("t", params, [f"s{i}" for i in range(8)], 40)
        added = [c for c in mixed if "screenshot-" in c]
        self.assertEqual(len(added), 2)  # about one per 20 seconds
        self.assertEqual(params.video_concat_mode, VideoConcatMode.sequential)

    def test_pipeline_mix_without_images_is_a_no_op(self):
        params = VideoParams(video_subject="x", add_site_screenshots=True)
        with mock.patch.object(task.utils, "task_dir", return_value=self.tmp):
            self.assertEqual(task._mix_in_screenshots("t", params, ["s0"], 30), ["s0"])
        self.assertNotEqual(params.video_concat_mode, VideoConcatMode.sequential)


class TestIntroPortrait(BrandingFolder):
    def test_vertical_video_uses_portrait_intro(self):
        Image.new("RGB", (1920, 1080), (255, 0, 0)).save(os.path.join(self.brand, "intro.png"))
        Image.new("RGB", (1080, 1920), (0, 255, 0)).save(os.path.join(self.brand, "intro_portrait.png"))
        final = os.path.join(self.tmp, "final.mp4")
        subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=blue:s=720x1280:d=2:r=30", "-f", "lavfi", "-i",
                        "sine=frequency=440:sample_rate=44100", "-shortest", "-pix_fmt", "yuv420p", final],
                       check=True)
        branding.add_intro_outro(final)
        with VideoFileClip(final) as clip:
            pixel = clip.get_frame(1.5)[640, 360]
        self.assertGreater(pixel[1], 200)
        self.assertLess(pixel[0], 60)


if __name__ == "__main__":
    unittest.main()
