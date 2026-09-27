import os
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from moviepy import VideoFileClip
from PIL import Image

from app.config import config
from app.models.schema import VideoAspect, VideoConcatMode, VideoParams, VideoTransitionMode
from app.services import branding, subtitle_styles, video
from app.utils import utils

FFMPEG = utils.get_ffmpeg_binary()


def _ffmpeg(*args):
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], check=True)


def _color_video(path, color, seconds, size="640x360", audio=False):
    args = ["-f", "lavfi", "-i", f"color=c={color}:s={size}:d={seconds}:r=30"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-shortest"]
    _ffmpeg(*args, "-pix_fmt", "yuv420p", path)


def _frame_rgb(path, t):
    with VideoFileClip(path) as clip:
        frame = clip.get_frame(t)
    return frame[frame.shape[0] // 2, frame.shape[1] // 2].astype(int)


def _frame_rgb_at(path, t, y):
    with VideoFileClip(path) as clip:
        frame = clip.get_frame(t)
    return frame[y, frame.shape[1] // 2].astype(int)


class TestResolution(unittest.TestCase):
    def test_default_is_1080p(self):
        with mock.patch.dict(config.app, {"video_resolution": ""}):
            self.assertEqual(VideoAspect.landscape.to_resolution(), (1920, 1080))
            self.assertEqual(VideoAspect.portrait.to_resolution(), (1080, 1920))

    def test_720p(self):
        with mock.patch.dict(config.app, {"video_resolution": "720p"}):
            self.assertEqual(VideoAspect.landscape.to_resolution(), (1280, 720))
            self.assertEqual(VideoAspect.portrait.to_resolution(), (720, 1280))
            self.assertEqual(VideoAspect.square.to_resolution(), (720, 720))


class TestSubtitleStyles(unittest.TestCase):
    def test_every_style_uses_a_bundled_font(self):
        for name, style in subtitle_styles.SUBTITLE_STYLES.items():
            self.assertTrue(
                os.path.isfile(os.path.join(utils.font_dir(), style["font_name"])), name
            )

    def test_apply_style(self):
        params = VideoParams(video_subject="x", subtitle_style="youtube_yellow")
        self.assertTrue(subtitle_styles.apply_subtitle_style(params))
        self.assertEqual(params.text_fore_color, "#FFD60A")
        self.assertEqual(params.font_name, "Tajawal-ExtraBold.ttf")

    def test_unknown_or_empty_style_keeps_custom_settings(self):
        params = VideoParams(video_subject="x", text_fore_color="#123456")
        self.assertFalse(subtitle_styles.apply_subtitle_style(params))
        params.subtitle_style = "does-not-exist"
        self.assertFalse(subtitle_styles.apply_subtitle_style(params))
        self.assertEqual(params.text_fore_color, "#123456")


class TestCrossfade(unittest.TestCase):
    def test_previous_shot_dissolves_into_next(self):
        self._check_crossfade(fast=False)

    def test_crossfade_with_ffmpeg_clip_preparation(self):
        self._check_crossfade(fast=True)

    def _check_crossfade(self, fast):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            config.app,
            {"video_resolution": "720p", "fast_clip_preparation": fast, "video_encode_preset": "veryfast"},
        ):
            red = os.path.join(tmp, "red.mp4")
            blue = os.path.join(tmp, "blue.mp4")
            audio = os.path.join(tmp, "a.mp3")
            out = os.path.join(tmp, "combined.mp4")
            _color_video(red, "red", 4)
            _color_video(blue, "blue", 4)
            _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "4", audio)
            video.combine_videos(
                combined_video_path=out,
                video_paths=[red, blue],
                audio_file=audio,
                video_aspect=VideoAspect.landscape,
                video_concat_mode=VideoConcatMode.sequential,
                video_transition_mode=VideoTransitionMode.crossfade,
                max_clip_duration=2,
                threads=1,
            )
            start = _frame_rgb(out, 0.5)
            blend = _frame_rgb(out, 2.2)
            after = _frame_rgb(out, 3.5)
            with VideoFileClip(out) as clip:
                self.assertAlmostEqual(clip.duration, 4.0, delta=0.15)
            self.assertEqual(clip.size, [1280, 720])
        self.assertGreater(start[0], 200)
        self.assertLess(start[2], 60)
        # Mid-dissolve both colours are visible.
        self.assertGreater(blend[0], 50)
        self.assertGreater(blend[2], 50)
        self.assertLess(after[0], 60)
        self.assertGreater(after[2], 200)


class TestCrossfadeAtSourceEnd(unittest.TestCase):
    def test_previous_shot_ending_with_its_source_does_not_drop_clips(self):
        # Regression: the tail past the end of a source used to be an empty
        # file, which made every following clip fail.
        for fast in (True, False):
            with self.subTest(fast=fast), tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                config.app, {"video_resolution": "720p", "fast_clip_preparation": fast}
            ):
                red = os.path.join(tmp, "red.mp4")
                blue = os.path.join(tmp, "blue.mp4")
                audio = os.path.join(tmp, "a.mp3")
                out = os.path.join(tmp, "combined.mp4")
                _color_video(red, "red", 2)  # shorter than max clip duration
                _color_video(blue, "blue", 6)
                _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "6", audio)
                with mock.patch.object(video.logger, "error") as error_log:
                    video.combine_videos(
                        combined_video_path=out,
                        video_paths=[red, blue],
                        audio_file=audio,
                        video_aspect=VideoAspect.landscape,
                        video_concat_mode=VideoConcatMode.sequential,
                        video_transition_mode=VideoTransitionMode.crossfade,
                        max_clip_duration=4,
                        threads=1,
                    )
                error_log.assert_not_called()
                blend = _frame_rgb(out, 2.2)  # red still frame dissolving into blue
                self.assertGreater(blend[0], 40)
                self.assertGreater(blend[2], 40)
                self.assertGreater(_frame_rgb(out, 4.5)[2], 200)


class TestFastClipPreparation(unittest.TestCase):
    def test_segment_is_cut_scaled_and_sped_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.mp4")
            out = os.path.join(tmp, "seg.mp4")
            _color_video(src, "red", 6, size="1920x1080")
            ok = video._prepare_segment_with_ffmpeg(
                src, 1.0, 5.0, 720, 1280, "cover", 2.0, out
            )
            self.assertTrue(ok)
            with VideoFileClip(out) as clip:
                self.assertEqual(clip.size, [720, 1280])
                self.assertAlmostEqual(clip.duration, 2.0, delta=0.1)
                self.assertIsNone(clip.audio)

    def test_contain_mode_letterboxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.mp4")
            out = os.path.join(tmp, "seg.mp4")
            _color_video(src, "white", 2, size="1920x1080")
            self.assertTrue(
                video._prepare_segment_with_ffmpeg(src, 0, 2, 720, 1280, "contain", 1.0, out)
            )
            top = _frame_rgb_at(out, 0.5, 20)
            self.assertTrue(all(c < 30 for c in top))  # black bar
            self.assertTrue(all(c > 220 for c in _frame_rgb(out, 0.5)))

    def test_failure_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(
                video._prepare_segment_with_ffmpeg(
                    os.path.join(tmp, "missing.mp4"), 0, 2, 720, 1280, "cover", 1.0,
                    os.path.join(tmp, "out.mp4"),
                )
            )

    def test_x264_preset_setting(self):
        with mock.patch.dict(config.app, {"video_encode_preset": "VeryFast"}):
            self.assertEqual(video._get_x264_preset(), "veryfast")
            self.assertEqual(video._with_x264_preset("libx264", {}), {"preset": "veryfast"})
            self.assertEqual(video._with_x264_preset("h264_nvenc", {}), {})
        with mock.patch.dict(config.app, {"video_encode_preset": "bogus"}):
            self.assertEqual(video._get_x264_preset(), "")


class TestBranding(unittest.TestCase):
    def test_find_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "intro.PNG"), "wb").close()
            open(os.path.join(tmp, "outro.txt"), "wb").close()
            self.assertTrue(branding.find_part("intro", tmp).endswith("intro.PNG"))
            self.assertEqual(branding.find_part("outro", tmp), "")

    def test_add_intro_image_and_silent_outro_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            brand = os.path.join(tmp, "branding")
            os.makedirs(brand)
            Image.new("RGB", (400, 400), (0, 255, 0)).save(os.path.join(brand, "intro.png"))
            _color_video(os.path.join(brand, "outro.mp4"), "white", 2, size="320x240")
            final = os.path.join(tmp, "final-1.mp4")
            _color_video(final, "blue", 4, size="1280x720", audio=True)

            branding.add_intro_outro(final, directory=brand)

            with VideoFileClip(final) as clip:
                self.assertAlmostEqual(clip.duration, 3 + 4 + 2, delta=0.3)
                self.assertEqual(clip.size, [1280, 720])
                self.assertIsNotNone(clip.audio)
            self.assertGreater(_frame_rgb(final, 1.5)[1], 200)  # green intro
            self.assertGreater(_frame_rgb(final, 5.0)[2], 200)  # blue main video
            self.assertTrue(all(c > 200 for c in _frame_rgb(final, 8.0)))  # white outro
            self.assertEqual(sorted(os.listdir(tmp)), ["branding", "final-1.mp4"])

    def test_missing_files_leave_video_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = os.path.join(tmp, "final.mp4")
            with open(final, "wb") as f:
                f.write(b"data")
            self.assertEqual(branding.add_intro_outro(final, directory=tmp), final)
            with open(final, "rb") as f:
                self.assertEqual(f.read(), b"data")


if __name__ == "__main__":
    unittest.main()
