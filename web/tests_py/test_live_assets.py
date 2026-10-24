import base64
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from web.tools import gen_live_assets as gla
from web.tools import pack_recordings as pr

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
CACHE = gla.DEFAULT_CACHE
ASSETS = CACHE / "assets"
LIVE_IN = CACHE / "live"
NAMES = ("g1", "x2", "t2")
STAND = json.loads((REPO / "deploy/vr_teleop/gmr_configs/stand.json").read_text())

_tmp = None
_out = None
_summary = None


def setUpModule():
  global _tmp, _out, _summary
  if not (ASSETS / "g1.xml").exists():
    raise unittest.SkipTest("web/.cache/assets not found (record_remote.sh fetch)")
  if not (LIVE_IN / "policy" / "policy.onnx").exists():
    raise unittest.SkipTest("web/.cache/live not found (tools/fetch_live_inputs.sh)")
  _tmp = tempfile.TemporaryDirectory()
  _out = Path(_tmp.name)
  _summary = gla.build(CACHE, _out)


def tearDownModule():
  if _tmp:
    _tmp.cleanup()


def live(name):
  return _out / "live" / name


def single(name):
  return mujoco.MjModel.from_xml_path(str(ASSETS / f"{name}.xml"))


def leak_strings():
  text = (WEB / "tools" / "leaks.mjs").read_text()
  block = text[text.index("const ENCODED"):text.index("];")]
  return [base64.b64decode(s).decode() for s in re.findall(r"'([^']+)'", block)]


def collides(ct_a, ca_a, ct_b, ca_b):
  return bool((ct_a & ca_b) or (ct_b & ca_a))


class LiveSceneTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = mujoco.MjModel.from_xml_path(str(live("scene.xml")))
    cls.scene = json.loads(live("scene.json").read_text())
    cls.robots = {r["name"]: r for r in cls.scene["robots"]}

  def robot_of_geom(self, g):
    """Robot name of geom g, or None for the floor (a world geom)."""
    body = self.model.body(self.model.geom_bodyid[g]).name
    return body.split("/")[0] if "/" in body else None

  def test_compiles_self_contained(self):
    m = self.model
    self.assertEqual([r["name"] for r in self.scene["robots"]], list(NAMES))
    self.assertEqual(m.nq, sum(single(n).nq for n in NAMES))
    self.assertEqual(m.nu, sum(single(n).nu for n in NAMES))
    self.assertNotIn("file=", live("scene.xml").read_text())  # meshes inline, no mesh files
    world = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
    self.assertEqual(len(world), 1)  # one floor
    self.assertEqual(m.geom_type[world[0]], mujoco.mjtGeom.mjGEOM_PLANE)
    self.assertEqual(m.nsensor, 0)
    self.assertEqual(m.nkey, 0)

  def test_physics_options_match_single(self):
    names = ("timestep", "gravity", "integrator", "cone", "solver", "jacobian", "iterations",
             "tolerance", "ls_iterations", "ls_tolerance", "ccd_iterations", "ccd_tolerance",
             "impratio", "noslip_iterations", "noslip_tolerance", "disableflags", "enableflags",
             "o_margin", "o_solref", "o_solimp", "o_friction", "density", "viscosity", "wind")
    for n in NAMES:
      s = single(n)
      for a in names:
        np.testing.assert_array_equal(getattr(self.model.opt, a), getattr(s.opt, a), err_msg=f"{n} {a}")
    robot = json.loads((REPO / "deploy/sim2sim/robots/g1.json").read_text())
    self.assertEqual(self.scene["timestep"], self.model.opt.timestep)
    self.assertEqual(self.scene["timestep"], robot["timing"]["physics_dt"])
    self.assertEqual(self.scene["decimation"], robot["timing"]["decimation"])
    self.assertAlmostEqual(self.scene["timestep"] * self.scene["decimation"], 0.02)  # 50 Hz

  def test_layout(self):
    m = self.model
    for n in NAMES:
      r = self.robots[n]
      s = single(n)
      baked = json.loads((REPO / f"deploy/sim2sim/robots/{n}.json").read_text())
      self.assertEqual(r["timestep"], baked["timing"]["physics_dt"])
      self.assertEqual(r["decimation"], baked["timing"]["decimation"])
      free = m.joint(f"{n}/robot/floating_base_joint")
      self.assertEqual(r["qposadr"], int(free.qposadr[0]))
      self.assertEqual(r["qveladr"], int(free.dofadr[0]))
      self.assertEqual((r["nq"], r["nv"], r["nu"]), (s.nq, s.nv, s.nu))
      self.assertAlmostEqual(m.qpos0[r["qposadr"] + 1], pr.LATERAL[n])
      self.assertEqual(r["lateral"], pr.LATERAL[n])
      # Joint maps: native joint i at qpos jointQposadr[i], dof jointDofadr[i].
      self.assertEqual(r["jointNames"], baked["native_joint_names"])
      for i, jn in enumerate(baked["native_joint_names"]):
        j = m.joint(f"{n}/robot/{jn}")
        self.assertEqual(r["jointQposadr"][i], int(j.qposadr[0]))
        self.assertEqual(r["jointDofadr"][i], int(j.dofadr[0]))
      # Actuators: ctrl[ctrladr + a] drives native joint actNative[a], in the single model's order.
      for a in range(s.nu):
        ca = r["ctrladr"] + a
        self.assertEqual(m.actuator(ca).name, f"{n}/{s.actuator(a).name}")
        jn = m.joint(int(m.actuator_trnid[ca, 0])).name
        self.assertEqual(jn, f"{n}/robot/{baked['native_joint_names'][r['actNative'][a]]}")
      self.assertEqual(r["imuBody"], f"{n}/robot/{baked['imu_body']}")
      self.assertEqual(m.body(r["imuBody"]).id, r["imuBodyId"])
      if r["imuSite"] is not None:
        self.assertEqual(m.site_bodyid[m.site(r["imuSite"]).id], r["imuBodyId"])
      self.assertEqual([m.body(b).name for b in r["trackedBodyIds"]],
                       [f"{n}/robot/{b}" for b in baked["tracked_bodies"]])

  def test_pd_gains_and_ctrl_ranges_match_single(self):
    m = self.model
    for n in NAMES:
      s = single(n)
      r = self.robots[n]
      sl = slice(r["ctrladr"], r["ctrladr"] + s.nu)
      for field in ("actuator_gainprm", "actuator_biasprm", "actuator_gaintype", "actuator_biastype",
                    "actuator_dyntype", "actuator_ctrlrange", "actuator_ctrllimited",
                    "actuator_forcerange", "actuator_forcelimited", "actuator_gear"):
        np.testing.assert_array_equal(getattr(m, field)[sl], getattr(s, field), err_msg=f"{n} {field}")
      self.assertEqual(r["ctrlRange"], s.actuator_ctrlrange.tolist())
      self.assertEqual(r["kp"], s.actuator_gainprm[:, 0].tolist())
      dv = slice(r["qveladr"], r["qveladr"] + s.nv)
      for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(m, field)[dv], getattr(s, field), err_msg=f"{n} {field}")
      for i in range(1, s.nbody):
        name = s.body(i).name
        if name == "terrain":
          continue
        b = m.body(f"{n}/{name}").id
        # Equal, or to the 6 significant digits the XML writer keeps (inertia from geoms: X2 pelvis)
        np.testing.assert_allclose(m.body_mass[b], s.body_mass[i], rtol=1e-5)
        np.testing.assert_allclose(m.body_inertia[b], s.body_inertia[i], rtol=1e-5)
      for j in range(s.njnt):
        cj = m.joint(f"{n}/{s.joint(j).name}").id
        np.testing.assert_array_equal(m.jnt_range[cj], s.jnt_range[j])
        self.assertEqual(m.jnt_limited[cj], s.jnt_limited[j])

  def test_collision_geoms_match_single(self):
    """Same collision geoms per robot, in the same places: every hull's support in 64
    directions agrees with the single model's at the stand pose."""
    m = self.model
    dirs = np.random.default_rng(0).normal(size=(64, 3))
    for n in NAMES:
      s = single(n)
      r = self.robots[n]
      sd, cd = mujoco.MjData(s), mujoco.MjData(m)
      sd.qpos[:] = r["standQpos"]
      sd.qpos[1] -= r["lateral"]
      cd.qpos[r["qposadr"]:r["qposadr"] + r["nq"]] = r["standQpos"]
      mujoco.mj_kinematics(s, sd)
      mujoco.mj_kinematics(m, cd)
      coll = [g for g in range(s.ngeom) if (s.geom_contype[g] or s.geom_conaffinity[g])
              and s.body(s.geom_bodyid[g]).name != "terrain"]
      ours = [g for g in range(m.ngeom) if self.robot_of_geom(g) == n]
      self.assertEqual(sorted(s.geom(g).name for g in coll),
                       sorted(m.geom(g).name[len(n) + 1:] for g in ours))
      for g in coll:
        cg = m.geom(f"{n}/{s.geom(g).name}").id
        self.assertEqual(m.geom_type[cg], s.geom_type[g])
        for f in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_priority",
                  "geom_margin", "geom_gap"):
          np.testing.assert_array_equal(getattr(m, f)[cg], getattr(s, f)[g])
        if s.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
          def support(model, data, gid):
            mid = model.geom_dataid[gid]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            w = data.geom_xpos[gid] + v @ data.geom_xmat[gid].reshape(3, 3).T
            return (w @ dirs.T).max(0)
          off = np.array([0.0, r["lateral"], 0.0]) @ dirs.T
          # within the 6 significant digits the XML keeps of the hull vertices and placement
          np.testing.assert_allclose(support(m, cd, cg), support(s, sd, g) + off, atol=1e-5)
        else:
          np.testing.assert_allclose(m.geom_size[cg], s.geom_size[g])
          np.testing.assert_allclose(cd.geom_xpos[cg], sd.geom_xpos[g] + [0, r["lateral"], 0], atol=1e-9)

  def test_robots_never_collide_pairwise(self):
    m = self.model
    owner = [self.robot_of_geom(g) for g in range(m.ngeom)]
    floor = owner.index(None)
    n_self = 0
    for a in range(m.ngeom):
      if owner[a] is None:
        continue
      ct_a, ca_a = m.geom_contype[a], m.geom_conaffinity[a]
      self.assertTrue(collides(ct_a, ca_a, m.geom_contype[floor], m.geom_conaffinity[floor]))
      for b in range(a + 1, m.ngeom):
        if owner[b] is None:
          continue
        c = collides(ct_a, ca_a, m.geom_contype[b], m.geom_conaffinity[b])
        if owner[a] != owner[b]:
          self.assertFalse(c, f"{m.geom(a).name} / {m.geom(b).name}")
        else:
          self.assertTrue(c)  # self-collision filtering is as in the single model (all 1/1 there)
          n_self += 1
    self.assertGreater(n_self, 1000)
    self.assertEqual(m.nexclude, single("g1").nexclude + single("x2").nexclude + single("t2").nexclude)

  def _overlapped(self, model):
    """All three robots in their stands at the same spot, stepped 20 times: the contacts."""
    d = mujoco.MjData(model)
    for r in self.scene["robots"]:
      q = np.array(r["standQpos"])
      q[1] -= r["lateral"]  # all at y = 0: the bodies interpenetrate
      d.qpos[r["qposadr"]:r["qposadr"] + r["nq"]] = q
      d.ctrl[r["ctrladr"]:r["ctrladr"] + r["nu"]] = self._stand_ctrl(r)
    pairs = set()
    for _ in range(20):
      mujoco.mj_step(model, d)
      for c in d.contact[:d.ncon]:
        pairs.add(frozenset((self.robot_of_geom(c.geom1), self.robot_of_geom(c.geom2))))
    return pairs

  def _stand_ctrl(self, r):
    q = np.array(r["standQpos"])
    return [q[adr - r["qposadr"]] for adr in np.array(r["jointQposadr"])[r["actNative"]]]

  def test_overlapping_robots_have_no_contact(self):
    pairs = self._overlapped(self.model)
    cross = {p for p in pairs if None not in p and len(p) == 2}
    self.assertEqual(cross, set())
    self.assertLessEqual({frozenset((None, n)) for n in NAMES}, pairs)  # the feet touch the floor
    # Control: with every geom colliding with every other, the same poses do touch.
    leaky = mujoco.MjModel.from_xml_path(str(live("scene.xml")))
    for a in ("geom_contype", "geom_conaffinity", "body_contype", "body_conaffinity"):
      getattr(leaky, a)[getattr(leaky, a) > 0] = 1  # the body masks gate the broad phase
    cross = {p for p in self._overlapped(leaky) if None not in p and len(p) == 2}
    self.assertTrue(cross)

  def test_dynamics_match_single(self):
    """Each robot under PD from its stand, in the combined scene and alone: the same motion."""
    m = self.model
    cd = mujoco.MjData(m)
    sims = {}
    for r in self.scene["robots"]:
      s = single(r["name"])
      sd = mujoco.MjData(s)
      q = np.array(r["standQpos"])
      cd.qpos[r["qposadr"]:r["qposadr"] + r["nq"]] = q
      q[1] -= r["lateral"]
      sd.qpos[:] = q
      sims[r["name"]] = (s, sd)
    t = 0.0
    for step in range(100):
      for r in self.scene["robots"]:
        s, sd = sims[r["name"]]
        ctrl = np.array(self._stand_ctrl(r)) + 0.1 * np.sin(2 * np.pi * t + np.arange(r["nu"]))
        cd.ctrl[r["ctrladr"]:r["ctrladr"] + r["nu"]] = ctrl
        sd.ctrl[:] = ctrl
        mujoco.mj_step(s, sd)
      mujoco.mj_step(m, cd)
      t += m.opt.timestep
    for r in self.scene["robots"]:
      s, sd = sims[r["name"]]
      got = cd.qpos[r["qposadr"]:r["qposadr"] + r["nq"]].copy()
      got[1] -= r["lateral"]
      np.testing.assert_allclose(got, sd.qpos, atol=1e-5, err_msg=r["name"])

  def test_stand_qpos_is_sim_node_stand(self):
    from deploy.sim2sim.obs import Robot
    from deploy.sim2sim.sim_node import SimRobot, stand_qpos

    for n in NAMES:
      sim = SimRobot(ASSETS / f"{n}.xml", Robot.load(n))
      want = stand_qpos(sim.model, sim, STAND[n])
      want[1] += pr.LATERAL[n]
      np.testing.assert_allclose(self.robots[n]["standQpos"], want, atol=1e-12)
    self.assertEqual(self.scene["fall"], {"height": 0.3, "tiltDeg": 60})


class LivePolicyFilesTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.doc = json.loads(live("robots.json").read_text())

  def test_no_absolute_paths_or_leaks(self):
    text = live("robots.json").read_text()
    self.assertNotIn("source_checkpoint", text)

    def strings(v):
      if isinstance(v, dict):
        for k, x in v.items():
          yield k
          yield from strings(x)
      elif isinstance(v, list):
        for x in v:
          yield from strings(x)
      elif isinstance(v, str):
        yield v

    for s in strings(self.doc):
      self.assertFalse(s.startswith(("/", "~")) or ":\\" in s or ".pt" in s, s)
    leaks = leak_strings()
    self.assertGreater(len(leaks), 5)
    for name in ("robots.json", "scene.json", "scene.xml", "clips.json"):
      blob = live(name).read_text()
      for s in leaks:
        self.assertNotIn(s, blob, f"{name}: {s!r}")

  def test_policy_fields(self):
    d = self.doc
    self.assertEqual((d["run"], d["iteration"]), ("unified_v2", 76000))
    self.assertEqual(d["historyLen"], 10)
    self.assertEqual((d["proprioDim"], d["goalK"], d["referenceLead"], d["fps"]), (93, 10, 1, 50))
    self.assertEqual(d["kpIdx"], [0, 3, 6, 10, 13])
    self.assertEqual((d["loop"]["leadInS"], d["loop"]["returnS"], d["loop"]["holdS"]), (2.0, 1.0, 1.0))
    self.assertEqual([r["name"] for r in d["robots"]], list(NAMES))
    for r in d["robots"]:
      n = r["name"]
      exp = json.loads((LIVE_IN / "policy" / f"{n}.json").read_text())
      baked = json.loads((REPO / f"deploy/sim2sim/robots/{n}.json").read_text())
      self.assertEqual(r["kpGoalVersion"], exp["kp_goal_version"])
      for k, ours in (("obs_mean", "obsMean"), ("obs_var", "obsVar"), ("goal_mean", "goalMean"),
                      ("goal_var", "goalVar"), ("kp_mean", "kpMean"), ("kp_var", "kpVar"),
                      ("morph", "morph")):
        self.assertEqual(r["inputs"][ours], exp["inputs"][k], k)
      src = exp["robot"]
      for b in (src, baked):  # the export's robot entry and the baked constants agree
        self.assertEqual(r["jointNames"], b["native_joint_names"])
        self.assertEqual(r["defaultJointPos"], b["default_joint_pos"])
        self.assertEqual(r["actionScale"], b["action_scale"])
        self.assertEqual((r["perm"], r["sign"], r["offset"]),
                         (b["translator"]["perm"], b["translator"]["sign"], b["translator"]["offset"]))
        self.assertEqual(r["imuBody"], b["imu_body"])
        self.assertEqual(r["trackedBodies"], b["tracked_bodies"])
        fk = b["fk"]
        self.assertEqual(r["fk"]["parent"], [x["parent"] for x in fk["bodies"]])
        self.assertEqual(r["fk"]["quat"], [x["quat"] for x in fk["bodies"]])
        self.assertEqual(r["fk"]["joints"], [[{"native": j["native_index"], "axis": j["axis"], "ref": j["ref"]}
                                              for j in x["joints"]] for x in fk["bodies"]])
        self.assertEqual((r["fk"]["slot0"], r["fk"]["anchor"], r["fk"]["imu"]),
                         (fk["slot0"], fk["anchor"], fk["imu"]))
      self.assertEqual(r["stand"]["state"], STAND[n]["state"])
      self.assertEqual(len(r["stand"]["state"]), 240)

  def test_policy_onnx_is_the_export(self):
    a = hashlib.sha256(live("policy.onnx").read_bytes()).hexdigest()
    b = hashlib.sha256((LIVE_IN / "policy" / "policy.onnx").read_bytes()).hexdigest()
    self.assertEqual(a, b)

  def test_policy_size_and_hash_recorded(self):
    blob = live("policy.onnx").read_bytes()
    self.assertEqual(self.doc["policy"], {"file": "policy.onnx", "bytes": len(blob),
                                          "sha256": hashlib.sha256(blob).hexdigest()})
    self.assertGreater(self.doc["policy"]["bytes"], 50_000_000)


class LiveClipsTest(unittest.TestCase):
  def test_clips_match_npz(self):
    doc = json.loads(live("clips.json").read_text())
    self.assertEqual(doc["stateDim"], 240)
    self.assertEqual(doc["columns"], 156)  # the goals read nothing past the body orientations
    cols = doc["columns"]
    self.assertEqual([(c["id"], c["label"]) for c in doc["clips"]], list(pr.CLIPS))
    for c in doc["clips"]:
      blob = live(f"clips/{c['id']}.bin").read_bytes()
      end = 0
      for n in NAMES:
        e = c["robots"][n]
        states = np.load(LIVE_IN / "clips" / n / f"{c['id']}.npz")["states"]
        self.assertEqual(e["frames"], states.shape[0])
        got = np.frombuffer(blob, "<f4", count=e["frames"] * cols, offset=e["offset"]).reshape(-1, cols)
        np.testing.assert_array_equal(got, states[:, :cols].astype(np.float32))
        end = max(end, e["offset"] + 4 * cols * e["frames"])
      self.assertEqual(end, len(blob))

  def test_total_size_reported(self):
    files = sorted(p for p in (_out / "live").rglob("*") if p.is_file())
    sizes = {str(p.relative_to(_out / "live")): p.stat().st_size for p in files}
    self.assertEqual(_summary["files"], sizes)
    self.assertEqual(_summary["total_bytes"], sum(sizes.values()))
    print(f"\nlive assets: {len(sizes)} files, {_summary['total_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
  unittest.main()
