"""Regression: assembling failed on Windows with FFmpeg 8:

    ffmpeg failed: Unrecognized option 'filter_complex_script'.
    Error splitting the argument list: Option not found

FFmpeg 8 removed ``-filter_complex_script``. Assembly must work with FFmpeg 6, 7 and 8.
"""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from moviepy import VideoFileClip

from app.services.presenter import assemble
from app.utils import utils

REAL_FFMPEG = utils.get_ffmpeg_binary()

SHIM = """#!{python}
import subprocess, sys
REMOVED = {removed!r}
for arg in sys.argv[1:]:
    if arg in REMOVED:
        sys.stderr.write("Unrecognized option '%s'.\\nError splitting the argument list: Option not found\\n" % arg[1:])
        sys.exit(8)
sys.exit(subprocess.call([{real!r}] + sys.argv[1:]))
"""


def fake_ffmpeg(folder, removed):
    path = os.path.join(folder, "ffmpeg")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(SHIM.format(python=sys.executable, removed=list(removed), real=REAL_FFMPEG))
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


@unittest.skipIf(os.name == "nt", "the fake ffmpeg is a POSIX script")
class TestFfmpegVersions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clips = []
        for i, color in enumerate(("red", "blue", "green")):
            clip = os.path.join(self.tmp, f"c{i}.mp4")
            subprocess.run([REAL_FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            f"color=c={color}:s=320x180:r=25:d=2", "-pix_fmt", "yuv420p", clip], check=True)
            self.clips.append(clip)

    def _join(self, removed, limit=assemble.INLINE_FILTER_LIMIT):
        out = os.path.join(self.tmp, f"out_{len(os.listdir(self.tmp))}.mp4")
        with mock.patch.dict(os.environ, {"IMAGEIO_FFMPEG_EXE": fake_ffmpeg(self.tmp, removed)}), \
                mock.patch.object(assemble, "INLINE_FILTER_LIMIT", limit):
            assemble.join_clips(self.clips, [1.5, 1.5, 1.5], out, 320, 180)
        with VideoFileClip(out) as clip:
            self.assertAlmostEqual(clip.duration, 4.5, delta=0.1)

    def test_ffmpeg8_without_filter_complex_script(self):
        self._join(["-filter_complex_script"])

    def test_long_graph_is_split_without_filter_files(self):
        # A very long video (many scenes) with FFmpeg 8 and with an FFmpeg that lacks "-/filter_complex".
        self._join(["-filter_complex_script", "-/filter_complex"], limit=0)
        self.assertFalse([f for f in os.listdir(self.tmp) if ".part" in f])  # halves cleaned up

    def test_real_errors_are_not_hidden(self):
        with mock.patch.dict(os.environ, {"IMAGEIO_FFMPEG_EXE": fake_ffmpeg(self.tmp, [])}):
            with self.assertRaises(RuntimeError):
                assemble.join_clips([os.path.join(self.tmp, "missing.mp4")], [1.0], os.path.join(self.tmp, "x.mp4"),
                                    320, 180)


if __name__ == "__main__":
    unittest.main()
