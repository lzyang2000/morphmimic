"""Dev-box side of the sim2sim: what the CPU simulation and the robot-side policy need,
taken from the training code (needs mjlab and a GPU; nothing else in
`deploy/sim2sim/` imports mjlab or torch).

For each robot it builds the eval env (`morphmimic_env_cfg(eval=True)`: no domain
randomisation, floor friction 1.0, no actuator delay) and writes:

- `deploy/sim2sim/robots/<robot>.json` (`--bake`): the robot constants of `obs.Robot`:
  native joint order, default pose, action scale, the canonical joint map, the
  kinematic tree from the root to the slot-0 and anchor bodies, the control timing,
  the PD gains, and for DC-motor joints (the X2's) the torque-speed curve, which
  `sim_node.py` applies to the explicit PD it computes for them. Committed;
  `test_sim2sim.py` re-derives it.
- `<out>/<robot>.xml` (meshes in `<out>/meshes/<robot>/`): the env's scene as plain,
  readable MJCF: the robot with the actuators, armature, floor, friction and solver
  options training used, which the robot's own MJCF does not hold. The export reloads
  it with plain `mujoco` and checks it against the env's compiled model
  (`check_same_model`); `sim_node.py` runs it.
- `<out>/clips/<robot>/<clip>.npz`: the clip's reference states [T, 240] (canonical,
  motion library frame, as the training goals read them), the robot's start state
  (qpos, qvel) as the eval env resets it, and a native qpos per reference frame for
  drawing the reference ghost.

  uv run python -m deploy.sim2sim.export_assets --robot g1 --subset-yaml <yaml> \\
    --clips walk_hands_on_back_loop_002__A096 wave_R_001__A430 --out <dir> [--bake]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np

from deploy.hardware.pose_source import SimPoseSource
from deploy.sim2sim.obs import ROBOTS_DIR, Robot

_PREFIX = "robot/"


def _body_id(m: mujoco.MjModel, name: str) -> int:
  i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, _PREFIX + name)
  if i < 0:
    raise ValueError(f"body {name} not in the model")
  return i


def fk_tree(m: mujoco.MjModel, native: list[str], root_body: str, slot0_body: str,
            anchor_body: str, imu_body: str, prefix: str = _PREFIX) -> dict:
  """The kinematic tree of `obs.Robot` (`fk`): the root body and every body between it
  and the slot-0, anchor and IMU bodies, root first, each with its parent, its
  position and orientation in the parent and its hinge joints (`native` index, axis,
  reference angle). Raises ValueError for a hinge off its body's origin (`jnt_pos` not
  zero): the tree has no joint offsets (none on the G1, X2 or T2)."""

  def body(name: str) -> int:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + name)
    if i < 0:
      raise ValueError(f"body {name} not in the model")
    return i

  root, slot0, anchor, imu = (body(b) for b in (root_body, slot0_body, anchor_body, imu_body))

  def chain(b: int) -> list[int]:
    path = [b]
    while path[-1] != root:
      path.append(int(m.body_parentid[path[-1]]))
      if path[-1] == 0:
        raise ValueError(f"body {b} is not below the root body")
    return path[::-1]

  order: list[int] = []
  for b in chain(slot0) + chain(anchor) + chain(imu):
    if b not in order:
      order.append(b)
  bodies = []
  for b in order:
    joints = []
    for j in range(m.njnt):
      if m.jnt_bodyid[j] != b or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        continue
      if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError("only hinge joints below the root are supported")
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).removeprefix(prefix)
      if np.any(m.jnt_pos[j] != 0.0):
        # `obs.Robot`'s forward kinematics rotates each body about its own origin.
        raise ValueError(f"hinge {name} sits at {m.jnt_pos[j].tolist()} in its body, not"
                         " at the body origin; the kinematic tree supports origin hinges"
                         " only")
      joints.append({
        "name": name,
        "native_index": native.index(name),
        "axis": m.jnt_axis[j].tolist(),
        "ref": float(m.qpos0[m.jnt_qposadr[j]]),
      })
    bodies.append({
      "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).removeprefix(prefix),
      "parent": -1 if b == root else order.index(int(m.body_parentid[b])),
      "pos": [0.0, 0.0, 0.0] if b == root else m.body_pos[b].tolist(),
      "quat": [1.0, 0.0, 0.0, 0.0] if b == root else m.body_quat[b].tolist(),
      "joints": joints,
    })
  return {"bodies": bodies, "slot0": order.index(slot0), "anchor": order.index(anchor),
          "imu": order.index(imu)}


def bake(env, robot: str) -> dict:
  """The `obs.Robot` constants of the eval env `env` of `robot`."""
  from morphmimic.constants import JOINT_NAMES
  from morphmimic.robots import get_robot

  spec = get_robot(robot)
  m = env.sim.mj_model
  art = env.scene["robot"]
  native = [str(n) for n in art.joint_names]
  term = env.action_manager.get_term("joint_pos")
  scale = np.broadcast_to(np.asarray(term._scale.cpu() if hasattr(term._scale, "cpu")
                                     else term._scale, np.float64).reshape(-1)[-len(native):],
                          (len(native),))
  tr = spec.translator

  fk = fk_tree(m, native, spec.root_body, spec.tracked_bodies[0], spec.anchor_body,
               spec.imu_body)

  pd = actuator_constants(spec, native)
  for a in range(m.nu):  # The built-in PD the compiled model holds must be the configs'.
    if m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
      continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
    i = native.index(name.removeprefix(_PREFIX))
    compiled = (m.actuator_gainprm[a, 0], -m.actuator_biasprm[a, 2], m.actuator_forcerange[a, 1])
    if not np.allclose(compiled, (pd["kp"][i], pd["kd"][i], pd["effort_limit"][i])):
      raise ValueError(f"{robot} {name}: compiled PD {compiled} differs from its config")

  out = {
    "robot": robot,
    "about": "Robot constants for the sim2sim and deployment (deploy/sim2sim/obs.py)."
             " Generated by deploy/sim2sim/export_assets.py --bake; do not edit.",
    "native_joint_names": native,
    "default_joint_pos": art.data.default_joint_pos[0].cpu().numpy().tolist(),
    "action_scale": scale.tolist(),
    "canonical_joint_names": list(JOINT_NAMES),
    "translator": {"perm": list(tr.perm), "sign": list(tr.sign), "offset": list(tr.offset)},
    "tracked_bodies": list(spec.tracked_bodies),
    "root_body": spec.root_body,
    "anchor_body": spec.anchor_body,
    "imu_body": spec.imu_body,
    "fk": fk,
    "timing": {"physics_dt": float(m.opt.timestep), "decimation": int(env.cfg.decimation)},
    "pd": {k: pd[k] for k in ("kp", "kd", "effort_limit")},
  }
  if "dc_motor" in pd:
    out["dc_motor"] = pd["dc_motor"]
  return out


def actuator_constants(spec, native: list[str]) -> dict:
  """Each native joint's PD gains and effort limit from the robot's actuator configs,
  and, when any joint is a DC motor (mjlab `DcMotorActuatorCfg`: explicit PD clipped to
  the torque-speed curve), each joint's stall torque and no-load speed (0 for the
  built-in PD joints) and the curve's law: "mjlab" (`DcMotorActuator`, full braking
  torque at any speed) or "booster" (`BoosterMotorActuatorCfg`, the bound falls with
  |v| in both directions); one law per robot."""
  from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg

  from morphmimic.robots.booster_motor import BoosterMotorActuatorCfg

  actuators = spec.entity_cfg_fn().articulation.actuators
  n = len(native)
  kp, kd, effort, sat, vmax = (np.zeros(n) for _ in range(5))
  laws = set()
  for i, name in enumerate(native):
    hits = [a for a in actuators if any(re.fullmatch(e, name) for e in a.target_names_expr)]
    if len(hits) != 1:
      raise ValueError(f"{spec.name} joint {name!r} matches {len(hits)} actuator configs")
    a = hits[0]
    if not isinstance(a, (BuiltinPositionActuatorCfg, DcMotorActuatorCfg)):
      raise TypeError(f"{spec.name} joint {name!r}: unsupported actuator {type(a).__name__}")
    kp[i], kd[i], effort[i] = a.stiffness, a.damping, a.effort_limit
    if isinstance(a, DcMotorActuatorCfg):
      sat[i], vmax[i] = a.saturation_effort, a.velocity_limit
      laws.add("booster" if isinstance(a, BoosterMotorActuatorCfg) else "mjlab")
  if len(laws) > 1:
    raise ValueError(f"{spec.name}: DC motors with both torque-speed laws")
  out = {"kp": kp.tolist(), "kd": kd.tolist(), "effort_limit": effort.tolist()}
  if vmax.any():
    out["dc_motor"] = {
      "saturation_effort": sat.tolist(),
      "velocity_limit": vmax.tolist(),
      "law": laws.pop(),
    }
  return out


# Compiled-model fields the readable XML must reproduce exactly (`check_same_model`).
SAME_MODEL_FIELDS = (
  "nq", "nv", "nu", "nbody", "njnt", "ngeom",
  "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos",
  "jnt_range", "jnt_axis", "qpos0", "dof_armature", "dof_damping", "dof_frictionloss",
  "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
  "actuator_trnid", "geom_friction", "geom_size", "geom_contype", "geom_conaffinity",
  "geom_condim", "geom_solref", "geom_solimp",
)
SAME_OPTION_FIELDS = (
  "timestep", "integrator", "iterations", "ls_iterations", "cone", "solver", "gravity",
)


_INTEGRATOR = ("Euler", "RK4", "implicit", "implicitfast")
_CONE = ("pyramidal", "elliptic")
_SOLVER = ("PGS", "CG", "Newton")
_JACOBIAN = ("dense", "sparse", "auto")


def _option_xml(opt) -> str:
  """An MJCF <option> element with the compiled model's solver settings, which mjlab
  sets on the compiled model (`MujocoCfg.apply`) rather than in the spec."""
  if opt.disableflags or opt.enableflags:
    raise ValueError("solver flags are set; extend _option_xml to write them")
  g = " ".join(f"{x:.9g}" for x in opt.gravity)
  return (
    f'  <option timestep="{opt.timestep:.9g}" gravity="{g}"'
    f' integrator="{_INTEGRATOR[opt.integrator]}" cone="{_CONE[opt.cone]}"'
    f' solver="{_SOLVER[opt.solver]}" jacobian="{_JACOBIAN[opt.jacobian]}"'
    f' iterations="{opt.iterations}" tolerance="{opt.tolerance:.9g}"'
    f' ls_iterations="{opt.ls_iterations}" ls_tolerance="{opt.ls_tolerance:.9g}"'
    f' ccd_iterations="{opt.ccd_iterations}" impratio="{opt.impratio:.9g}"/>\n'
  )


def write_scene_xml(env, robot: str, out: Path) -> Path:
  """The env's scene as standalone MJCF `<out>/<robot>.xml`, its meshes copied to
  `<out>/meshes/<robot>/`."""
  from mjlab.utils.spec import fix_spec_xml, strip_buffer_textures

  from morphmimic.robots import get_robot

  spec = env.scene.spec.copy()
  strip_buffer_textures(spec)
  xml = fix_spec_xml(spec.to_xml(), meshdir=f"meshes/{robot}")
  child = get_robot(robot).entity_cfg_fn().spec_fn()
  src = Path(child.modelfiledir) / child.meshdir
  mesh_dir = out / "meshes" / robot
  mesh_dir.mkdir(parents=True, exist_ok=True)
  for name in sorted(set(re.findall(r'<mesh [^>]*file="([^"]+)"', xml))):
    shutil.copy2(src / name, mesh_dir / Path(name).name)
  if "<option" in xml:
    raise ValueError("the scene XML already has an <option> element")
  # The solver settings training used, right after the <compiler> element.
  compiler_end = xml.index("/>", xml.index("<compiler")) + 3
  xml = xml[:compiler_end] + _option_xml(env.sim.mj_model.opt) + xml[compiler_end:]
  path = out / f"{robot}.xml"
  path.write_text(xml)
  return path


def check_same_model(trained: mujoco.MjModel, loaded: mujoco.MjModel) -> list[str]:
  """Names of the `SAME_MODEL_FIELDS` / `SAME_OPTION_FIELDS` that differ, up to the
  XML's 6 significant digits."""
  bad = []
  for f in SAME_MODEL_FIELDS:
    a, b = np.asarray(getattr(trained, f)), np.asarray(getattr(loaded, f))
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-5, atol=1e-8):
      bad.append(f)
  for f in SAME_OPTION_FIELDS:
    if not np.allclose(getattr(trained.opt, f), getattr(loaded.opt, f)):
      bad.append(f"opt.{f}")
  return bad


def check_fk(model: mujoco.MjModel, robot: Robot, trials: int = 20) -> float:
  """Largest error between MuJoCo's slot-0 and anchor orientations (rad) and anchor
  position (m) and those `Robot` computes from the IMU body's orientation, the joints
  and (the position) the free body's position and yaw, over random poses."""
  rng = np.random.default_rng(0)
  data = mujoco.MjData(model)
  slot0 = _body_id(model, robot.tracked_bodies[0])
  anchor = _body_id(model, robot.tracked_bodies[7])
  imu = _body_id(model, robot.imu_body)
  qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _PREFIX + n)]
          for n in robot.native_joint_names]
  worst = 0.0
  for _ in range(trials):
    data.qpos[:] = model.qpos0
    root_quat = rng.normal(size=4)
    data.qpos[3:7] = root_quat / np.linalg.norm(root_quat)
    q = rng.uniform(-0.8, 0.8, size=robot.num_native)
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    for body, est in ((anchor, robot.anchor_quat(data.xquat[imu].copy(), q)),
                      (slot0, robot.slot0_quat(data.xquat[imu].copy(), q))):
      dot = abs(float(np.dot(est, data.xquat[body])))
      worst = max(worst, 2.0 * np.arccos(min(1.0, dot)))
    pos, _ = robot.anchor_pose(*SimPoseSource(model, data).pose(), data.xquat[imu].copy(), q)
    worst = max(worst, float(np.abs(pos - data.xpos[anchor]).max()))
  return worst


def export_clips(env, robot: str, yaml_path: str, clips: list[str], out: Path) -> list[Path]:
  """`<out>/clips/<robot>/<clip>.npz` for each clip (see the module docstring)."""
  import torch

  from morphmimic.robots import get_robot
  from morphmimic.scripts.play import ClipPlayback, resolve_clip

  spec = get_robot(robot)
  m = env.sim.mj_model
  native = [str(n) for n in env.scene["robot"].joint_names]
  default = env.scene["robot"].data.default_joint_pos[0].cpu()
  qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, _PREFIX + n)])
          for n in native]
  origin = env.scene.env_origins[0].cpu().numpy()
  root_slot = spec.tracked_bodies.index(spec.root_body)
  written = []
  for clip in clips:
    pb = ClipPlayback(env, resolve_clip(yaml_path, clip), hist_len=10)
    library = pb.command.library
    start = int(library.clip_start[pb.clip_id])
    frames = torch.arange(start, start + pb.clip_len, device=env.device)
    states = library.state(frames).float().cpu()
    qpos = env.sim.data.qpos[0].cpu().numpy().astype(np.float64)
    qvel = env.sim.data.qvel[0].cpu().numpy().astype(np.float64)
    qpos[:3] -= origin
    ghost = np.tile(m.qpos0, (len(states), 1))
    from morphmimic.model.state import split_state

    parts = split_state(states)
    ghost[:, :3] = parts.body_pos[:, root_slot].numpy()
    ghost[:, 3:7] = parts.body_quat[:, root_slot].numpy()
    ghost[:, qadr] = spec.translator.to_native(parts.joint_pos, default).numpy()
    path = out / "clips" / robot / f"{pb.clip_name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, states=states.numpy().astype(np.float32), init_qpos=qpos,
             init_qvel=qvel, ghost_qpos=ghost, clip=pb.clip_name, robot=robot)
    written.append(path)
  return written


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--robot", required=True, choices=["g1", "x2", "t2", "g1_amber"])
  parser.add_argument("--subset-yaml", required=True)
  parser.add_argument("--clips", nargs="+", required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--bake", action="store_true", help="also write robots/<robot>.json")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
  os.environ.setdefault("MUJOCO_GL", "egl")

  from morphmimic.env.env_cfg import morphmimic_env_cfg
  from morphmimic.env.morphmimic_env import MorphMimicEnv

  cfg = morphmimic_env_cfg(args.robot, subset_yaml=args.subset_yaml, num_envs=1, eval=True)
  env = MorphMimicEnv(cfg, "cuda:0")
  try:
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = write_scene_xml(env, args.robot, args.out)
    loaded = mujoco.MjModel.from_xml_path(str(model_path))
    bad = check_same_model(env.sim.mj_model, loaded)
    if bad:
      raise ValueError(f"{model_path} differs from the trained model in: {bad}")
    constants = bake(env, args.robot)
    if args.bake:
      (ROBOTS_DIR / f"{args.robot}.json").write_text(json.dumps(constants, indent=1) + "\n")
    robot = Robot.load(args.robot)
    if robot.native_joint_names != tuple(constants["native_joint_names"]):
      raise ValueError(f"robots/{args.robot}.json is stale: re-run with --bake")
    err = check_fk(loaded, robot)
    print(f"[export] {args.robot}: model {model_path}; FK from the IMU max error {err:.2e} (rad, m)")
    if err > 1e-5:
      raise ValueError(f"{args.robot}: anchor FK does not match MuJoCo ({err:.2e} rad or m)")
    for path in export_clips(env, args.robot, args.subset_yaml, args.clips, args.out):
      print(f"[export] {args.robot}: {path}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
