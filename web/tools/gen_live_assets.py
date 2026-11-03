"""Build the live demo's physics scene, policy inputs and reference clips.

  uv run --project .. --group export python tools/gen_live_assets.py [--cache DIR] [--out DIR]

Reads <cache>/assets/{g1_amber,x2,t2}.xml (the exported robot scenes) and <cache>/live/ (from
tools/fetch_live_inputs.sh: policy/{policy.onnx,<robot>.json}, clips/<robot>/<clip>.npz), plus
deploy/sim2sim/robots/<robot>.json and deploy/vr_teleop/gmr_configs/stand.json. Writes
<out>/live/:

  scene.xml    one MuJoCo model for the physics: each robot's bodies, joints, actuators (the
               built-in position PD of its exported scene), contact excludes and collision geoms,
               prefixed g1/ x2/ t2/, placed at the lateral offsets of the recordings (X2 -1.5,
               G1 0, T2 +1.5 m), and one floor. Visual geoms, sensors, keyframes, lights and
               materials are left out; collision meshes are written inline as the vertices of
               their convex hulls (what MuJoCo collides), and every body keeps the inertia its
               single-robot model compiled to. Robot i's geoms have contype = conaffinity = bit
               i + 1 and the floor all three bits, so a robot touches the floor and itself as in
               its own scene, never another robot.
  scene.json   the robot side (deploy/sim2sim/sim_node.py): physics timestep and decimation,
               the fall check, and per robot its qpos/qvel/ctrl blocks, native joint -> qpos and
               dof addresses, actuator -> native joint, ctrl ranges, the DC motors (`dc`: per
               actuator whether it is a torque actuator and the constants of the explicit PD
               and torque-speed curve SimRobot.dc_torque computes for it; null without), the
               IMU body (its orientation and gyro, as sim_node reads them) and the qpos of the
               neutral stand.
  robots.json  the policy side (deploy/sim2sim/obs.py, policy_node.py): per robot the ONNX
               normalisers and morphology vector, joint maps, default pose, action scale, the
               kinematic tree for the anchor and slot-0 orientations, and the stand state;
               and `policy` {file, bytes, sha256} of policy.onnx (the page's download progress
               uses bytes when the server sends no content-length).
  policy.onnx  the shared policy cut down to its motion output (`action_motion`) and the
               inputs that reads (policy_node.pruned): the page runs the motion interface only.
  clips.json   {fps, stateDim: 240, columns: 156, clips: [{id, label, robots: {<robot>:
               {offset, frames}}}]} in sidebar order, and
  clips/<id>.bin  per clip the robots' reference states, float32 little-endian
               [frames, columns], at the byte offsets in clips.json: only the first 156 of the
               240 state columns (joints, joint velocities, body positions and orientations),
               the ones the goals read; the body velocities (156-239) are left out and the
               page fills them with zeros (src/live/reference.js statesFromClip).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import warnings
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
if str(REPO) not in sys.path:  # Run as a script: `deploy` and `web` live at the repo root.
  sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from web.tools.gen_scene import ROBOTS, freeze_inertials  # noqa: E402
from web.tools.pack_recordings import CLIPS, FPS, LATERAL  # noqa: E402

DEFAULT_CACHE = WEB / ".cache"
DEFAULT_OUT = WEB / "public"
BAKED_DIR = REPO / "deploy" / "sim2sim" / "robots"
STAND_FILE = REPO / "deploy" / "vr_teleop" / "gmr_configs" / "stand.json"

# The page's robot (prefix in the scene, clip folder) -> the robot the policy export and the
# exported scene name: the G1 slot is the G1 body with the released policy's actuators.
POLICY_ROBOT = {"g1": "g1_amber"}

STATE_DIM = 240
# The state columns the goals read (obs.js): everything before the body velocities (LIN = 156).
CLIP_COLUMNS = 156
# The cue loop (CuedClipReference): lead in from the stand, the clip, back to the stand, hold.
LEAD_IN_S = 2.0
HOLD_S = 1.0
FLOOR_BITS = sum(1 << (i + 1) for i in range(len(ROBOTS)))
OPTION_FIELDS = ("timestep", "gravity", "integrator", "cone", "solver", "jacobian", "iterations",
                 "tolerance", "ls_iterations", "ls_tolerance", "ccd_iterations", "ccd_tolerance",
                 "impratio", "noslip_iterations", "noslip_tolerance", "disableflags",
                 "enableflags", "o_margin", "o_solref", "o_solimp", "o_friction", "density",
                 "viscosity", "wind", "magnetic")
FLOOR_FIELDS = ("type", "size", "friction", "condim", "solref", "solimp", "solmix", "priority",
                "margin", "gap")


def hull_vertices(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
  """The vertices of a compiled mesh's convex hull (all its vertices if it has no hull), in the
  mesh frame the model placed its geoms with."""
  v0, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
  verts = model.mesh_vert[v0:v0 + vn]
  adr = model.mesh_graphadr[mesh_id]
  if adr < 0:
    return verts.copy()
  n = model.mesh_graph[adr]
  return verts[model.mesh_graph[adr + 2 + n:adr + 2 + 2 * n]].copy()


def physics_spec(path: Path, bit: int) -> tuple[mujoco.MjSpec, mujoco.MjSpec]:
  """A robot's exported scene reduced to what its physics needs, its geoms in contact group
  `bit`; and the scene as loaded (for its options and floor)."""
  model = mujoco.MjSpec.from_file(str(path)).compile()
  spec = mujoco.MjSpec.from_file(str(path))
  freeze_inertials(spec, model)
  geoms = list(spec.geoms)
  if [g.name for g in geoms] != [model.geom(i).name for i in range(model.ngeom)]:
    raise ValueError(f"{path}: spec and model geom order differ")
  terrain = [b for b in spec.bodies if b.name == "terrain"]
  if len(terrain) != 1:
    raise ValueError(f"{path}: expected one terrain body")
  doomed, mesh_of = [], {}
  for i, g in enumerate(geoms):
    if model.body(model.geom_bodyid[i]).name == "terrain":
      continue
    if g.contype == 0 and g.conaffinity == 0:
      doomed.append(g)  # visual only
      continue
    if (g.contype, g.conaffinity) != (1, 1):
      raise ValueError(f"{path}: geom {g.name} has contype/conaffinity {g.contype}/{g.conaffinity}")
    g.contype = g.conaffinity = bit
    g.material = ""
    if g.type == mujoco.mjtGeom.mjGEOM_MESH:
      # The compiled placement holds the mesh's recentring; the hull vertices are in that frame.
      g.pos = model.geom_pos[i]
      g.quat = model.geom_quat[i]
      mesh_of.setdefault(g.meshname, int(model.geom_dataid[i]))
  for g in doomed:
    spec.delete(g)
  spec.delete(terrain[0])
  for mesh in list(spec.meshes):
    if mesh.name not in mesh_of:
      spec.delete(mesh)
      continue
    verts = hull_vertices(model, mesh_of[mesh.name])
    mesh.file = ""
    mesh.content_type = ""
    mesh.material = ""
    mesh.scale = [1.0, 1.0, 1.0]
    mesh.refpos = [0.0, 0.0, 0.0]
    mesh.refquat = [1.0, 0.0, 0.0, 0.0]
    mesh.uservert = verts.ravel().tolist()
    for attr in ("userface", "usernormal", "usertexcoord", "userfacenormal", "userfacetexcoord"):
      setattr(mesh, attr, [])
  for coll in (spec.sensors, spec.lights, spec.cameras, spec.materials, spec.textures):
    for item in list(coll):
      spec.delete(item)
  spec.nkey = 0
  spec.compile()  # attach() needs the child compiled, or its keyframes clash by name
  return spec, mujoco.MjSpec.from_file(str(path))


def _same(a, b) -> bool:
  return np.array_equal(np.asarray(a), np.asarray(b))


def build_scene(assets: Path) -> tuple[mujoco.MjSpec, mujoco.MjModel]:
  combined = mujoco.MjSpec()
  combined.compiler.degree = False
  parts = []
  for i, (name, _) in enumerate(ROBOTS):
    parts.append(physics_spec(assets / f"{POLICY_ROBOT.get(name, name)}.xml", 1 << (i + 1)))
  first = parts[0][1]
  for _, src in parts[1:]:
    for f in OPTION_FIELDS:
      if not _same(getattr(src.option, f), getattr(first.option, f)):
        raise ValueError(f"robot scenes differ in option {f}")
  for f in OPTION_FIELDS:
    setattr(combined.option, f, getattr(first.option, f))
  floors = [next(g for g in src.geoms if g.name == "terrain") for _, src in parts]
  for f in FLOOR_FIELDS:
    if not all(_same(getattr(g, f), getattr(floors[0], f)) for g in floors):
      raise ValueError(f"robot scenes differ in floor {f}")
  floor = combined.worldbody.add_geom()
  floor.name = "floor"
  for f in FLOOR_FIELDS:
    setattr(floor, f, getattr(floors[0], f))
  floor.contype = floor.conaffinity = FLOOR_BITS
  for (name, _), (spec, _) in zip(ROBOTS, parts):
    frame = combined.worldbody.add_frame()
    frame.pos = [0.0, LATERAL[name], 0.0]
    with warnings.catch_warnings():  # attach warns that child options (timestep...) are ignored
      warnings.simplefilter("ignore")
      combined.attach(spec, prefix=f"{name}/", frame=frame)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    combined.compile()  # brings in the robots' keyframes, which the scene does not use
    for key in list(combined.keys):
      combined.delete(key)
    model = combined.compile()
  return combined, model


def scene_meta(model: mujoco.MjModel, assets: Path) -> dict:
  from deploy.sim2sim.obs import Robot
  from deploy.sim2sim.sim_node import SimRobot, stand_qpos
  from deploy.vr_teleop.constants import FALL_HEIGHT, FALL_TILT_DEG

  stand = json.loads(STAND_FILE.read_text())
  robots, timing = [], set()
  for name, label in ROBOTS:
    robot = Robot.load(POLICY_ROBOT.get(name, name), BAKED_DIR)
    # checks the timestep against the constants, and finds the DC motors
    single = SimRobot(assets / f"{robot.name}.xml", robot)
    timing.add((robot.physics_dt, robot.decimation))
    p = f"{name}/robot/"
    free = model.joint(p + "floating_base_joint")
    joints = [model.joint(p + j) for j in robot.native_joint_names]
    acts = [a for a in range(model.nu) if model.actuator(a).name.startswith(f"{name}/")]
    if acts != list(range(acts[0], acts[0] + len(acts))) or len(acts) != single.model.nu:
      raise ValueError(f"{name}: actuators are not one block")
    jid = {int(j.id): i for i, j in enumerate(joints)}
    imu = model.body(p + robot.imu_body).id
    sites = [model.site(s).name for s in range(model.nsite)
             if model.site_bodyid[s] == imu and "imu" in model.site(s).name]
    q = stand_qpos(single.model, single, stand[robot.body])
    q[1] += LATERAL[name]
    robots.append({
        "name": name, "label": label, "lateral": LATERAL[name],
        "qposadr": int(free.qposadr[0]), "nq": single.model.nq,
        "qveladr": int(free.dofadr[0]), "nv": single.model.nv,
        "ctrladr": acts[0], "nu": len(acts),
        "jointNames": list(robot.native_joint_names),
        "jointQposadr": [int(j.qposadr[0]) for j in joints],
        "jointDofadr": [int(j.dofadr[0]) for j in joints],
        "actNative": [jid[int(model.actuator_trnid[a, 0])] for a in acts],
        "ctrlRange": model.actuator_ctrlrange[acts].tolist(),
        "dc": dc_meta(single),
        "imuBody": p + robot.imu_body, "imuBodyId": int(imu),
        "imuSite": sites[0] if sites else None,
        "trackedBodyIds": [int(model.body(p + b).id) for b in robot.tracked_bodies],
        "standQpos": q.tolist(),
        "timestep": robot.physics_dt, "decimation": robot.decimation,
    })
  if len(timing) != 1:
    raise ValueError(f"robots differ in timing: {timing}")
  dt, decimation = timing.pop()
  if not math.isclose(model.opt.timestep, dt):
    raise ValueError(f"scene timestep {model.opt.timestep}, robot constants {dt}")
  return {
      "timestep": dt, "decimation": decimation, "controlDt": dt * decimation,
      "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
      "fall": {"height": FALL_HEIGHT, "tiltDeg": FALL_TILT_DEG},
      "robots": robots,
  }


def dc_meta(single) -> dict | None:
  """SimRobot's DC motors per actuator (`motor` 0/1; kp, kd, limit, sat, vmax, 0 off the motors)
  and the torque-speed law; None when the robot has none."""
  if not single.motor.any():
    return None
  _, _, kp, kd, limit, sat, vmax = single._dc
  out = {"motor": single.motor.astype(int).tolist(), "law": single._dc_law}
  for key, v in (("kp", kp), ("kd", kd), ("limit", limit), ("sat", sat), ("vmax", vmax)):
    full = np.zeros(single.model.nu)
    full[single.motor] = v
    out[key] = full.tolist()
  return out


def policy_meta(policy_dir: Path) -> dict:
  import onnxruntime as ort

  from deploy.sim2sim.obs import BODIES, GOAL_K, PROPRIO_DIM
  from deploy.sim2sim.policy_node import REFERENCE_LEAD
  from deploy.vr_teleop.constants import STAND_BLEND_S

  session = ort.InferenceSession(str(policy_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
  width = {i.name: i.shape[-1] for i in session.get_inputs()}["obs_history"]
  if width % PROPRIO_DIM:
    raise ValueError(f"obs_history width {width} is not a multiple of {PROPRIO_DIM}")
  stand = json.loads(STAND_FILE.read_text())
  runs, robots = set(), []
  for name, _ in ROBOTS:
    policy_robot = POLICY_ROBOT.get(name, name)
    doc = json.loads((policy_dir / f"{policy_robot}.json").read_text())
    runs.add((doc["run"], doc["iteration"], doc["goal_version"]))
    if doc.get("goal_anchor", "reference") != "reference":
      raise ValueError(f"{name}: the page builds reference-anchored motion goals only")
    r, inp, fk = doc["robot"], doc["inputs"], doc["robot"]["fk"]
    robots.append({
        "name": name,
        "inputs": {"obsMean": inp["obs_mean"], "obsVar": inp["obs_var"],
                   "goalMean": inp["goal_mean"], "goalVar": inp["goal_var"],
                   "morph": inp["morph"]},
        "jointNames": r["native_joint_names"],
        "defaultJointPos": r["default_joint_pos"],
        "actionScale": r["action_scale"],
        "perm": r["translator"]["perm"], "sign": r["translator"]["sign"],
        "offset": r["translator"]["offset"],
        "imuBody": r["imu_body"],
        "trackedBodies": r["tracked_bodies"],
        "fk": {"parent": [b["parent"] for b in fk["bodies"]],
               "quat": [b["quat"] for b in fk["bodies"]],
               "joints": [[{"native": j["native_index"], "axis": j["axis"], "ref": j["ref"]}
                           for j in b["joints"]] for b in fk["bodies"]],
               "slot0": fk["slot0"], "anchor": fk["anchor"], "imu": fk["imu"]},
        "stand": {"state": stand[BODIES.get(policy_robot, policy_robot)]["state"]},
    })
  if len(runs) != 1:
    raise ValueError(f"robot exports differ in run: {runs}")
  run, iteration, goal_version = runs.pop()
  return {
      "run": run, "iteration": iteration, "goalVersion": goal_version,
      "proprioDim": PROPRIO_DIM, "historyLen": width // PROPRIO_DIM, "goalK": GOAL_K,
      "stateDim": STATE_DIM, "referenceLead": REFERENCE_LEAD,
      "fps": FPS,
      "loop": {"leadInS": LEAD_IN_S, "returnS": STAND_BLEND_S, "holdS": HOLD_S},
      "robots": robots,
  }


def pack_clips(clip_dir: Path, out: Path) -> dict:
  (out / "clips").mkdir(parents=True, exist_ok=True)
  clips = []
  for cid, label in CLIPS:
    blocks, entry, offset = [], {}, 0
    for name, _ in ROBOTS:
      states = np.load(clip_dir / name / f"{cid}.npz")["states"]
      if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"{name}/{cid}: states {states.shape}")
      b = np.ascontiguousarray(states[:, :CLIP_COLUMNS], dtype="<f4").tobytes()
      entry[name] = {"offset": offset, "frames": int(states.shape[0])}
      blocks.append(b)
      offset += len(b)
    (out / "clips" / f"{cid}.bin").write_bytes(b"".join(blocks))
    clips.append({"id": cid, "label": label, "robots": entry})
  return {"fps": FPS, "stateDim": STATE_DIM, "columns": CLIP_COLUMNS, "clips": clips}


def build(cache: Path = DEFAULT_CACHE, out: Path = DEFAULT_OUT) -> dict:
  cache, assets = Path(cache), Path(cache) / "assets"
  live_in = cache / "live"
  dest = Path(out) / "live"
  if dest.exists():
    shutil.rmtree(dest)
  dest.mkdir(parents=True)

  spec, model = build_scene(assets)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    (dest / "scene.xml").write_text(spec.to_xml())
  (dest / "scene.json").write_text(json.dumps(scene_meta(model, assets), indent=1))
  import onnx

  from deploy.sim2sim.policy_node import pruned

  (dest / "policy.onnx").write_bytes(
      pruned(onnx.load(live_in / "policy" / "policy.onnx"), "action_motion"))
  meta = policy_meta(live_in / "policy")
  onnx = (dest / "policy.onnx").read_bytes()
  meta["policy"] = {"file": "policy.onnx", "bytes": len(onnx), "sha256": hashlib.sha256(onnx).hexdigest()}
  (dest / "robots.json").write_text(json.dumps(meta))
  (dest / "clips.json").write_text(json.dumps(pack_clips(live_in / "clips", dest), indent=1))

  files = {str(p.relative_to(dest)): p.stat().st_size
           for p in sorted(dest.rglob("*")) if p.is_file()}
  return {"files": files, "total_bytes": sum(files.values()),
          "nq": int(model.nq), "nu": int(model.nu), "ngeom": int(model.ngeom)}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = ap.parse_args()
  s = build(args.cache, args.out)
  for name, size in s["files"].items():
    if not name.startswith("clips/"):
      print(f"  {name:12s} {size / 1e6:8.2f} MB")
  clips = sum(v for k, v in s["files"].items() if k.startswith("clips/"))
  print(f"  clips/       {clips / 1e6:8.2f} MB ({sum(k.startswith('clips/') for k in s['files'])} files)")
  print(f"total {s['total_bytes'] / 1e6:.2f} MB; scene nq {s['nq']} nu {s['nu']} geoms {s['ngeom']}")


if __name__ == "__main__":
  main()
