import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from web.tools import gen_scene

CACHE = gen_scene.DEFAULT_CACHE
ASSETS = CACHE / "assets"
NAMES = [n for n, _ in gen_scene.ROBOTS]

_tmp = None
_out = None
_summary = None


def setUpModule():
  global _tmp, _out, _summary
  if not (ASSETS / "g1.xml").exists():
    raise unittest.SkipTest("web/.cache/assets not found (run Task 2 export)")
  _tmp = tempfile.TemporaryDirectory()
  _out = Path(_tmp.name)
  _summary = gen_scene.build(CACHE, _out)


def tearDownModule():
  if _tmp:
    _tmp.cleanup()


def random_pose(model, rng):
  q = np.zeros(model.nq)
  for j in range(model.njnt):
    adr = model.jnt_qposadr[j]
    t = model.jnt_type[j]
    if t == mujoco.mjtJoint.mjJNT_FREE:
      q[adr:adr + 3] = rng.uniform(-1, 1, 3)
      quat = rng.normal(size=4)
      q[adr + 3:adr + 7] = quat / np.linalg.norm(quat)
    elif model.jnt_limited[j]:
      lo, hi = model.jnt_range[j]
      q[adr] = rng.uniform(lo, hi)
    else:
      q[adr] = rng.uniform(-1, 1)
  return q


class GenSceneTest(unittest.TestCase):
  def setUp(self):
    self.comb = mujoco.MjModel.from_xml_path(str(_out / "model" / "scene.xml"))
    self.scene = json.loads((_out / "model" / "scene.json").read_text())

  def test_combined_fk_matches_single_robot(self):
    rng = np.random.default_rng(0)
    for r in self.scene["robots"]:
      single = mujoco.MjModel.from_xml_path(str(ASSETS / f"{r['name']}.xml"))
      self.assertEqual(single.nq, r["nq"])
      q = random_pose(single, rng)
      cd, sd = mujoco.MjData(self.comb), mujoco.MjData(single)
      cd.qpos[r["qposadr"]:r["qposadr"] + r["nq"]] = q
      sd.qpos[:] = q
      mujoco.mj_kinematics(self.comb, cd)
      mujoco.mj_kinematics(single, sd)
      n = 0
      for i in range(1, single.nbody):
        name = single.body(i).name
        if name == "terrain":
          continue
        b = self.comb.body(f"{r['name']}/{name}")
        np.testing.assert_allclose(cd.xpos[b.id], sd.xpos[i], atol=1e-9)
        np.testing.assert_allclose(cd.xquat[b.id], sd.xquat[i], atol=1e-9)
        n += 1
      self.assertGreater(n, 25)

  def test_joint_order_matches_recording(self):
    for r in self.scene["robots"]:
      single = mujoco.MjModel.from_xml_path(str(ASSETS / f"{r['name']}.xml"))
      want = [single.joint(j).name for j in range(single.njnt)]
      prefix = r["name"] + "/"
      adr = r["qposadr"]
      got = [self.comb.joint(j).name[len(prefix):] for j in range(self.comb.njnt)
             if self.comb.joint(j).name.startswith(prefix)]
      self.assertEqual(got, want)
      self.assertEqual(self.comb.joint(prefix + want[0]).qposadr[0], adr)

  def test_robot_layout(self):
    self.assertEqual([r["name"] for r in self.scene["robots"]], ["g1", "x2", "t2"])
    self.assertEqual(self.scene["robots"][0]["qposadr"], 0)
    self.assertEqual(sum(r["nq"] for r in self.scene["robots"]), self.comb.nq)
    self.assertEqual(self.comb.ngeom, 0)

  def test_mesh_budget(self):
    self.assertLessEqual(_summary["triangles"], 250_000)
    self.assertLessEqual((_out / "model" / "meshes.bin").stat().st_size, 6 * 1024 * 1024)
    meta = json.loads((_out / "model" / "meshes.json").read_text())
    self.assertEqual(sum(m["faceCount"] for m in meta), _summary["triangles"])
    nv = sum(m["vertCount"] for m in meta)
    blob = (_out / "model" / "meshes.bin").read_bytes()
    self.assertEqual(len(blob), 12 * (nv + _summary["triangles"]))
    faces = np.frombuffer(blob, "<u4", offset=12 * nv).reshape(-1, 3)
    for m in meta:
      f = faces[m["faceOffset"]:m["faceOffset"] + m["faceCount"]]
      self.assertLess(int(f.max()), m["vertCount"])

  def test_kp_bodies(self):
    want = {
      "g1": ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link",
             "right_wrist_yaw_link"],
      "x2": ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link",
             "right_wrist_roll_link"],
      "t2": ["waist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_roll_link", "right_wrist_roll_link"],
    }
    kp = self.scene["kpBodies"]
    self.assertEqual(len(kp), 3)
    for r, idx in zip(self.scene["robots"], kp):
      self.assertEqual(len(idx), 5)
      names = [self.scene["bodies"][i] for i in idx]
      self.assertEqual(names, [f"{r['name']}/robot/{b}" for b in want[r["name"]]])
      for n in names:  # bodies[i] is MuJoCo body id i + 1 in the combined scene
        self.assertEqual(self.comb.body(n).id, self.scene["bodies"].index(n) + 1)

  def test_visuals_reference_known_bodies(self):
    bodies = set(self.scene["bodies"])
    visuals = json.loads((_out / "model" / "visuals.json").read_text())
    meshes = {m["name"] for m in json.loads((_out / "model" / "meshes.json").read_text())}
    self.assertGreater(len(visuals), 60)
    for v in visuals:
      self.assertIn(v["body"], bodies)
      self.assertIn(v["mesh"], meshes)
      self.assertEqual((len(v["pos"]), len(v["quat"]), len(v["rgba"])), (3, 4, 4))


class DecimateTest(unittest.TestCase):
  def test_cube_grid_keeps_shape(self):
    g = np.linspace(0, 1, 30)
    verts = np.array([[x, y, 0.0] for x in g for y in g])
    faces = []
    for i in range(29):
      for j in range(29):
        a = i * 30 + j
        faces += [[a, a + 30, a + 1], [a + 1, a + 30, a + 31]]
    v, f = gen_scene.decimate(verts, np.array(faces), 0.1)
    self.assertLess(len(f), len(faces))
    self.assertGreater(len(f), 0)
    self.assertLess(int(f.max()), len(v))


if __name__ == "__main__":
  unittest.main()
