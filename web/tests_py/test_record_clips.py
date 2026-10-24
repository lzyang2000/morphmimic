import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from web.tools import record_clips

ROOT = Path(__file__).resolve().parents[2]
# Directory of exported G1 assets (g1.xml, meshes/, clips/g1/<CLIP>.npz). Required: the test is
# skipped when SM_TEST_ASSETS is unset.
_ASSETS_ENV = os.environ.get("SM_TEST_ASSETS", "")
ASSETS = Path(_ASSETS_ENV).expanduser() if _ASSETS_ENV else None
# An export_onnx.py folder with g1.json (policy.onnx, <robot>.json). Defaults to
# deploy/ckpt/goal_v2/onnx when that exists; the test is skipped without either.
_DEPLOY_ENV = os.environ.get("SM_TEST_DEPLOY", "")
DEPLOY = (Path(_DEPLOY_ENV).expanduser() if _DEPLOY_ENV
          else ROOT / "deploy/ckpt/goal_v2/onnx")
CLIP = "walk_ff_loop_315_R_001__A257"
DROPPED = ("states", "ghost_qpos", "init_qpos")


@unittest.skipUnless(ASSETS is not None, "SM_TEST_ASSETS is not set (directory of exported G1 assets)")
@unittest.skipUnless((DEPLOY / "g1.json").exists(),
                     "no exported policy: set SM_TEST_DEPLOY to an export_onnx.py folder")
class RecordOneTest(unittest.TestCase):
  def setUp(self):
    if not (ASSETS / "g1.xml").exists():
      self.fail(f"SM_TEST_ASSETS={ASSETS} has no g1.xml")

  def test_record_one_returns_qpos_per_step(self):
    qpos, summary = record_clips.record_one(DEPLOY, ASSETS, "g1", CLIP, "motion")
    self.assertEqual(qpos.shape, (summary["frames"], 36))
    self.assertTrue(np.isfinite(qpos).all())
    self.assertEqual(summary["result"], "completed")
    # A plain clip run (not from the stand): one row per control step from frame 0, which is
    # the clip's first pose, and as many rows as the clip has frames.
    self.assertNotIn("mode", summary)
    ref = np.load(ASSETS / "clips/g1" / f"{CLIP}.npz")
    self.assertEqual(summary["frames"], len(ref["states"]))
    np.testing.assert_allclose(qpos[0], ref["init_qpos"], atol=1e-9)
    self.assertGreater(np.abs(qpos[1] - qpos[0]).max(), 0.0)
    for key in DROPPED:
      self.assertNotIn(key, summary)
    with tempfile.TemporaryDirectory() as out:
      record_clips.write_recording(Path(out), CLIP, "g1", "motion", qpos, summary)
      written = json.loads((Path(out) / CLIP / "g1_motion.json").read_text())
      for key in DROPPED:
        self.assertNotIn(key, written)
      self.assertEqual(written["nq"], 36)
      self.assertEqual(np.load(Path(out) / CLIP / "g1_motion.npy").shape, qpos.shape)


if __name__ == "__main__":
  unittest.main()
