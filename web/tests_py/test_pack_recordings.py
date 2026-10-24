import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from web.tools import pack_recordings as pr

CLIP = pr.CLIPS[0][0]
NQ = {"g1": 36, "x2": 38, "t2": 38}


def yaw_of(q):
  w, x, y, z = q
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def make_rec(frames, nq, seed, yaw0=0.7, xy0=(3.0, -2.0)):
  rng = np.random.default_rng(seed)
  q = np.zeros((frames, nq))
  q[:, 3] = 1.0
  q[:, 2] = 0.8
  walk = np.cumsum(rng.normal(0, 0.02, (frames, 2)), axis=0)
  q[:, 0:2] = np.array(xy0) + walk
  yaw = yaw0 + np.cumsum(rng.normal(0, 0.01, frames))
  q[:, 3] = np.cos(yaw / 2)
  q[:, 6] = np.sin(yaw / 2)
  q[:, 7:] = rng.normal(0, 0.3, (frames, nq - 7))
  return q


def write_fixture(cache, failed=()):
  rec = Path(cache) / "recordings" / CLIP
  rec.mkdir(parents=True)
  qs = {}
  for k, robot in enumerate(NQ):
    for j, iface in enumerate(("motion", "keypoint")):
      frames = 100 + 10 * k + j
      q = make_rec(frames, NQ[robot], 10 * k + j)
      stem = f"{robot}_{iface}"
      np.save(rec / f"{stem}.npy", q)
      result = "failed: fell at 3.2 s" if (robot, iface) in failed else "completed"
      (rec / f"{stem}.json").write_text(json.dumps({
          "robot": robot, "clip": CLIP, "frames": frames, "result": result,
          "joint_err_rms_rad": 0.05 + 0.01 * k, "interface": iface, "nq": NQ[robot],
          "states": [1, 2], "ghost_qpos": [1], "init_qpos": [1]}))
      qs[(robot, iface)] = q
  (Path(cache) / "policy.txt").write_text("unified_v2 76000\n")
  return qs


class AlignTest(unittest.TestCase):
  def test_align_faces_plus_x_at_lateral_offset(self):
    q = make_rec(50, 36, 1)
    a = pr.align(q, -1.5)
    self.assertAlmostEqual(yaw_of(a[0, 3:7]), 0.0, delta=1e-9)
    np.testing.assert_allclose(a[0, 0:2], [0.0, -1.5], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(a[:, 3:7], axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(a[:, 7:], q[:, 7:])
    np.testing.assert_allclose(a[:, 2], q[:, 2])

  def test_align_preserves_relative_motion(self):
    q = make_rec(50, 38, 2)
    a = pr.align(q, 1.5)
    d0 = np.linalg.norm(q[:, None, 0:3] - q[None, :, 0:3], axis=2)
    d1 = np.linalg.norm(a[:, None, 0:3] - a[None, :, 0:3], axis=2)
    np.testing.assert_allclose(d1, d0, atol=1e-9)
    # heading change relative to frame 0 is preserved too
    dy0 = np.unwrap([yaw_of(x) for x in q[:, 3:7]]) - yaw_of(q[0, 3:7])
    dy1 = np.unwrap([yaw_of(x) for x in a[:, 3:7]])
    np.testing.assert_allclose(dy1, dy0, atol=1e-9)

  def test_align_does_not_mutate_input(self):
    q = make_rec(10, 36, 3)
    before = q.copy()
    pr.align(q, 0.0)
    np.testing.assert_array_equal(q, before)


class PackTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.cache = Path(self.tmp.name) / "cache"
    self.out = Path(self.tmp.name) / "public"

  def tearDown(self):
    self.tmp.cleanup()

  def index(self):
    return json.loads((self.out / "rec" / "index.json").read_text())

  def test_pack_marks_failed_recordings_unavailable(self):
    write_fixture(self.cache, failed={("x2", "keypoint")})
    pr.pack(self.cache, self.out)
    idx = self.index()
    clip = idx["clips"][0]
    bad = clip["robots"]["x2"]["keypoint"]
    self.assertFalse(bad["available"])
    self.assertEqual(bad["frames"], 0)
    self.assertTrue(clip["robots"]["x2"]["motion"]["available"])
    size = (self.out / "rec" / f"{CLIP}.bin").stat().st_size
    ok_frames = sum(e["frames"] * e["nq"] for r in clip["robots"].values() for e in r.values())
    self.assertEqual(size, 4 * ok_frames)
    # clips with no recordings at all are listed as unavailable, with no bin
    self.assertFalse(idx["clips"][1]["robots"]["g1"]["motion"]["available"])
    self.assertFalse((self.out / "rec" / f"{idx['clips'][1]['id']}.bin").exists())

  def test_index_has_no_reference_fields(self):
    write_fixture(self.cache)
    pr.pack(self.cache, self.out)
    text = (self.out / "rec" / "index.json").read_text()
    for key in ("states", "ghost_qpos", "init_qpos", "init_qvel"):
      self.assertNotIn(key, text)

  def test_index_structure(self):
    write_fixture(self.cache)
    pr.pack(self.cache, self.out)
    idx = self.index()
    self.assertEqual(idx["policy"], "unified_v2 76000")
    self.assertEqual(idx["fps"], 50)
    self.assertEqual([c["id"] for c in idx["clips"]], [c for c, _ in pr.CLIPS])
    self.assertEqual(len(idx["clips"]), 10)
    clip = idx["clips"][0]
    self.assertEqual(clip["label"], "Walk")
    self.assertEqual(list(clip["robots"]), ["g1", "x2", "t2"])
    self.assertAlmostEqual(clip["duration"], (max(100 + 20 + 1, 0) - 1) / 50)
    e = clip["robots"]["t2"]["keypoint"]
    self.assertEqual(set(e), {"offset", "frames", "available", "jointErr", "nq"})
    self.assertEqual(e["nq"], 38)
    self.assertAlmostEqual(e["jointErr"], 0.07)

  def test_bin_roundtrip(self):
    qs = write_fixture(self.cache)
    pr.pack(self.cache, self.out)
    clip = self.index()["clips"][0]
    blob = (self.out / "rec" / f"{CLIP}.bin").read_bytes()
    lateral = {"g1": -1.5, "x2": 0.0, "t2": 1.5}
    for robot, ifaces in clip["robots"].items():
      for iface, e in ifaces.items():
        self.assertEqual(e["offset"] % 4, 0)
        got = np.frombuffer(blob, "<f4", count=e["frames"] * e["nq"], offset=e["offset"])
        want = pr.align(qs[(robot, iface)], lateral[robot])
        np.testing.assert_allclose(got.reshape(e["frames"], e["nq"]), want, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
