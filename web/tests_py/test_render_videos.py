import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from web.tools import render_videos

CACHE = render_videos.DEFAULT_CACHE


class CameraAndShrinkTest(unittest.TestCase):
  def test_group_camera_follows_the_mean_from_a_fixed_azimuth(self):
    import numpy as np
    n = 400
    recs = {}
    for name, y in (("g1", -1.5), ("x2", 0.0), ("t2", 1.5)):
      q = np.zeros((n, 36))
      q[:, 0] = np.linspace(0.0, 4.0, n)  # walking along x
      q[:, 1] = y
      q[:, 3:7] = (np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4))  # heading +y: ignored
      recs[name] = q
    cam = render_videos.camera_path(recs, n, distance=4.2, elevation=-14.0)
    np.testing.assert_allclose(cam[:, 3], render_videos.AZIMUTH)
    np.testing.assert_allclose(cam[:, 4:6], np.tile([-14.0, 4.2], (n, 1)))
    np.testing.assert_allclose(cam[:, 1], 0.0, atol=1e-12)  # the group's centre in y
    np.testing.assert_allclose(cam[:, 2], render_videos.LOOK_Z)
    self.assertEqual(cam[0, 0], 0.0)
    self.assertTrue(np.all(np.diff(cam[:, 0]) > 0))  # follows, smoothed: lags the mean
    self.assertTrue(np.all(cam[1:, 0] < recs["g1"][1:, 0]))
    self.assertAlmostEqual(cam[-1, 0], 4.0, delta=0.25)

  def test_shrink_warns_when_it_cannot_fit(self):
    with tempfile.TemporaryDirectory() as tmp:
      out = Path(tmp) / "noise.mp4"
      subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                      "testsrc=size=160x90:rate=10", "-t", "1", "-c:v", "libx264", "-crf", "30",
                      "-pix_fmt", "yuv420p", str(out)], check=True)
      with self.assertWarns(RuntimeWarning):
        crf = render_videos.shrink(out, 1, 30)
      self.assertGreaterEqual(crf, render_videos.MAX_CRF)


class HeroTest(unittest.TestCase):
  def test_hero_cut_from_the_clip_video(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp = Path(tmp)
      # A 15 s, 50 fps stand-in for the rendered ballet-turns video, with audio and a title tag.
      subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                      "testsrc=size=160x90:rate=50", "-f", "lavfi", "-i", "sine", "-t", "15",
                      "-metadata", "title=secret", "-c:v", "libx264", "-pix_fmt", "yuv444p",
                      "-c:a", "aac", str(tmp / f"{render_videos.HERO_CLIP}.mp4")], check=True)
      video, poster = render_videos.render_hero(tmp)
      info = json.loads(subprocess.run(
          ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
           "stream=codec_type,codec_name,pix_fmt,nb_read_frames:format=duration:format_tags",
           "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout)
      self.assertEqual([s["codec_type"] for s in info["streams"]], ["video"], "no audio")
      s = info["streams"][0]
      self.assertEqual((s["codec_name"], s["pix_fmt"]), ("h264", "yuv420p"))
      self.assertEqual(int(s["nb_read_frames"]), 495)  # 3.36 .. 13.26 s at 50 fps
      self.assertAlmostEqual(float(info["format"]["duration"]), 9.9, delta=0.03)
      tags = info["format"].get("tags", {})
      self.assertNotIn("title", tags)
      self.assertNotIn("encoder", tags)
      with open(video, "rb") as f:
        head = f.read(4096)
      self.assertIn(b"moov", head, "faststart: the index is at the front")
      from PIL import Image
      self.assertEqual(Image.open(poster).size, (160, 90))
      cmd = render_videos.hero_cmds(tmp / "x.mp4", tmp)[0]
      self.assertEqual(cmd[cmd.index("-crf") + 1], "23")
      self.assertEqual(cmd[cmd.index("-ss") + 1], "3.36")

  def test_hero_needs_the_clip_video(self):
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaises(FileNotFoundError):
        render_videos.render_hero(Path(tmp))


class RenderVideosTest(unittest.TestCase):
  def setUp(self):
    if not (CACHE / "assets" / "g1.xml").exists() or not (CACHE / "recordings").exists():
      self.skipTest("web/.cache assets/recordings not found")

  def test_render_short_clip(self):
    clip = render_videos.WALK  # one group shot like every other clip
    with tempfile.TemporaryDirectory() as tmp:
      out = Path(tmp) / "short.mp4"
      poster = Path(tmp) / "short_poster.jpg"
      n = render_videos.render_clip(clip, out, poster, frames=25)
      self.assertTrue(out.exists())
      self.assertTrue(poster.exists())
      info = json.loads(subprocess.run(
          ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,nb_read_frames,codec_name,pix_fmt",
           "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout)
      s = info["streams"][0]
      self.assertEqual((s["width"], s["height"]), (1280, 720))
      self.assertEqual((s["codec_name"], s["pix_fmt"]), ("h264", "yuv420p"))
      self.assertLessEqual(abs(int(s["nb_read_frames"]) - 25), 1)
      self.assertEqual(n, 25)
      from PIL import Image
      import numpy as np
      px = np.asarray(Image.open(poster).convert("RGB"))
      for corner in (px[:8, :8], px[:8, -8:]):  # sky: white, not MuJoCo's blue
        self.assertTrue((corner.reshape(-1, 3).mean(0) >= 235).all(), corner.mean((0, 1)))
      floor = px[-8:, :8].reshape(-1, 3).min(0)  # the floor is pure white, below the robots
      self.assertTrue((floor >= 245).all(), floor)
      band = px[int(px.shape[0] * 0.6):]  # floor rows: shadows under the feet stay visible
      self.assertLess(band.mean(2).min(), 235, band.mean(2).min())


if __name__ == "__main__":
  unittest.main()
