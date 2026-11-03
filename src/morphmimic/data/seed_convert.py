"""SEED-format robot CSV -> 50 Hz npz motions with full-body kinematics.

SEED CSV columns: Frame, root translation (cm), root rotation (extrinsic Euler xyz,
degrees), then one column per robot joint (degrees), sampled at 120 Hz. BONES-SEED G1
CSVs name the joint columns `<joint>_dof` in G1 order; NVIDIA's SOMA Retargeter writes
the X2 / T2 columns as `<joint>`; columns are mapped by name. The root is the robot's
free-joint body (`RobotSpec.root_body`). Each clip is converted to metres / radians /
wxyz quaternions, resampled to 50 Hz (lerp + slerp), run through MuJoCo forward
kinematics on CPU for all bodies of the robot, differentiated by finite differences and
saved as an npz with mjlab's motion keys plus `body_names`, `joint_names` (native
order), `robot`, `root_body` and the per-frame filter statistics `z_min`, `foot_dist`.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
import re
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from morphmimic.constants import FPS
from morphmimic.data.filters import feasibility_reject, name_filter_reject
from morphmimic.data.manifest import MANIFEST_COLUMNS
from morphmimic.robots import RobotSpec, get_robot

SEED_FPS = 120
SEED_ROOT_COLUMNS: tuple[str, ...] = (
  "Frame",
  "root_translateX",
  "root_translateY",
  "root_translateZ",
  "root_rotateX",
  "root_rotateY",
  "root_rotateZ",
)
FLOOR_GEOM = "floor"
# G1 foot collision geoms (`foot_geoms` of the G1 model with its foot regex).
FOOT_GEOMS: tuple[tuple[str, ...], tuple[str, ...]] = (
  tuple(f"left_foot{i}_collision" for i in range(1, 8)),
  tuple(f"right_foot{i}_collision" for i in range(1, 8)),
)
# Foot-floor distances are only computed up to this value (metres).
FOOT_DIST_MAX = 1.0


class FKResult(NamedTuple):
  """Forward kinematics of a clip; bodies in model order without the world body."""

  body_pos_w: np.ndarray  # [T, B, 3]
  body_quat_w: np.ndarray  # [T, B, 4] wxyz
  z_min: np.ndarray  # [T] lowest origin height of the included bodies
  foot_dist: np.ndarray  # [T, 2] signed left/right foot distance to the floor


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
  return q[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
  return q[..., [1, 2, 3, 0]]


def parse_csv(
  path: str, robot: RobotSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Reads a SEED-format CSV of `robot`.

  The header is SEED_ROOT_COLUMNS followed by one column per joint of `robot`, in any
  order, named `<joint>` or `<joint>_dof`.

  Returns:
    root_pos [T, 3] in metres, root_quat [T, 4] wxyz, joint_pos [T, J] in radians in
    `robot.joint_names` order.

  Raises:
    ValueError: if the root columns differ, or a joint column is missing, unknown or
      duplicated.
  """
  df = pd.read_csv(path)
  columns = list(df.columns)
  root = columns[: len(SEED_ROOT_COLUMNS)]
  if root != list(SEED_ROOT_COLUMNS):
    raise ValueError(
      f"{path}: unexpected root columns {root}, expected {list(SEED_ROOT_COLUMNS)}"
    )
  joint_columns = columns[len(SEED_ROOT_COLUMNS) :]
  by_joint: dict[str, str] = {}
  for column in joint_columns:
    joint = column.removesuffix("_dof")
    if joint not in robot.joint_names:
      joint = column
    if joint not in robot.joint_names:
      raise ValueError(f"{path}: column {column!r} is not a {robot.name} joint")
    if joint in by_joint:
      raise ValueError(
        f"{path}: joint {joint!r} has two columns, {by_joint[joint]!r} and {column!r}"
      )
    by_joint[joint] = column
  missing = [j for j in robot.joint_names if j not in by_joint]
  if missing:
    raise ValueError(f"{path}: missing {robot.name} joint columns {missing}")
  root_values = df[list(SEED_ROOT_COLUMNS)].to_numpy(dtype=np.float64)
  joint_values = df[[by_joint[j] for j in robot.joint_names]].to_numpy(np.float64)
  root_pos = root_values[:, 1:4] / 100.0
  root_rot = Rotation.from_euler("xyz", root_values[:, 4:7], degrees=True)
  root_quat = _xyzw_to_wxyz(root_rot.as_quat())
  joint_pos = np.deg2rad(joint_values)
  return root_pos, root_quat, joint_pos


def parse_seed_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Reads a BONES-SEED G1 CSV (`parse_csv` with the G1 spec)."""
  return parse_csv(path, get_robot("g1"))


def resample(
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  joint_pos: np.ndarray,
  in_fps: int,
  out_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Resamples to t_k = k / out_fps, k = 0..floor(duration * out_fps).

  Positions and joints are linearly interpolated, root quaternions (wxyz) slerped.
  """
  n_in = len(root_pos)
  duration = (n_in - 1) / in_fps
  n_out = int(np.floor(duration * out_fps + 1e-9)) + 1
  t_out = np.minimum(np.arange(n_out) / out_fps, duration)
  x = t_out * in_fps
  i0 = np.clip(np.floor(x).astype(np.int64), 0, max(n_in - 2, 0))
  i1 = np.minimum(i0 + 1, n_in - 1)
  w = (x - i0)[:, None]

  def lerp(a: np.ndarray) -> np.ndarray:
    return a[i0] * (1.0 - w) + a[i1] * w

  slerp = Slerp(
    np.arange(n_in) / in_fps, Rotation.from_quat(_wxyz_to_xyzw(root_quat))
  )
  quat = _xyzw_to_wxyz(slerp(t_out).as_quat())
  return lerp(root_pos), quat, lerp(joint_pos)


def make_model(robot: str = "g1") -> mujoco.MjModel:
  """The robot's mjlab MuJoCo model plus a floor plane at z = 0 (cached per robot, do
  not modify)."""
  return _make_model(robot)


@functools.cache  # Keyed by the robot name alone (make_model() is make_model("g1")).
def _make_model(robot: str) -> mujoco.MjModel:
  robot_spec = get_robot(robot)
  spec = robot_spec.entity_cfg_fn().spec_fn()
  spec.worldbody.add_geom(
    name=FLOOR_GEOM, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1]
  )
  model = spec.compile()
  joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
  if (
    model.nq != 7 + len(robot_spec.joint_names)
    or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
    or joint_names != robot_spec.joint_names
  ):
    raise RuntimeError(
      f"unexpected {robot} model layout: nq={model.nq}, joints={joint_names}"
    )
  return model


def body_names(model: mujoco.MjModel) -> tuple[str, ...]:
  """All model bodies except the world body, in model order (root body first)."""
  return tuple(model.body(i).name for i in range(1, model.nbody))


# The G1's bodies (pelvis first).
ENTITY_BODY_NAMES: tuple[str, ...] = body_names(make_model())


def foot_geoms(
  model: mujoco.MjModel, regex: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """(left, right) geom names fully matching `regex`, split by the left_/right_ prefix.

  Raises:
    ValueError: if a side has no geom or a match has neither prefix.
  """
  pattern = re.compile(regex)
  left: list[str] = []
  right: list[str] = []
  for i in range(model.ngeom):
    name = model.geom(i).name
    if not pattern.fullmatch(name):
      continue
    if name.startswith("left_"):
      left.append(name)
    elif name.startswith("right_"):
      right.append(name)
    else:
      raise ValueError(f"foot geom {name!r} has no left_/right_ prefix")
  if not left or not right:
    raise ValueError(f"no left or right foot geoms match {regex!r}")
  return tuple(left), tuple(right)


def zmin_bodies(model: mujoco.MjModel, exclude_regex: str | None) -> list[int]:
  """Ids of the bodies (world excluded) whose names do not fully match the regex."""
  pattern = re.compile(exclude_regex) if exclude_regex is not None else None
  return [
    i
    for i in range(1, model.nbody)
    if pattern is None or not pattern.fullmatch(model.body(i).name)
  ]


def forward_kinematics(
  model: mujoco.MjModel,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  joint_pos: np.ndarray,
  zmin_body_ids: list[int] | None = None,
  *,
  feet: tuple[tuple[str, ...], tuple[str, ...]],
) -> FKResult:
  """Body poses, lowest body height and foot-floor distances for each frame.

  Args:
    zmin_body_ids: bodies whose origins count for `z_min` (default all but the world).
    feet: (left, right) foot geom names for `foot_dist`, e.g.
      `foot_geoms(model, spec.foot_geom_regex)` (FOOT_GEOMS for the G1).
  """
  data = mujoco.MjData(model)
  floor = model.geom(FLOOR_GEOM).id
  feet_ids = [[model.geom(name).id for name in side] for side in feet]
  qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
  n, nb = len(qpos), model.nbody - 1
  body_pos = np.empty((n, nb, 3))
  body_quat = np.empty((n, nb, 4))
  foot_dist = np.empty((n, 2))
  for t in range(n):
    data.qpos[:] = qpos[t]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    body_pos[t] = data.xpos[1:]
    body_quat[t] = data.xquat[1:]
    for side, geoms in enumerate(feet_ids):
      foot_dist[t, side] = min(
        mujoco.mj_geomDistance(model, data, g, floor, FOOT_DIST_MAX, None)
        for g in geoms
      )
  zmin_cols = (
    slice(None) if zmin_body_ids is None else np.asarray(zmin_body_ids) - 1
  )
  z_min = body_pos[:, zmin_cols, 2].min(axis=1)
  return FKResult(body_pos, body_quat, z_min, foot_dist)


def _angular_velocity(
  q_from: np.ndarray, q_to: np.ndarray, dt: float
) -> np.ndarray:
  """World-frame angular velocity rotating wxyz q_from into q_to over dt."""
  shape = q_from.shape[:-1]
  r_from = Rotation.from_quat(_wxyz_to_xyzw(q_from).reshape(-1, 4))
  r_to = Rotation.from_quat(_wxyz_to_xyzw(q_to).reshape(-1, 4))
  return (r_to * r_from.inv()).as_rotvec().reshape(*shape, 3) / dt


def finite_diff_velocities(
  joint_pos: np.ndarray, body_pos_w: np.ndarray, body_quat_w: np.ndarray, fps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Central finite differences (one-sided at the ends).

  Angular velocity is the axis-angle of q_{t+1} q_{t-1}^{-1} over 2 dt (world frame).

  Returns:
    joint_vel [T, J], body_lin_vel_w [T, B, 3], body_ang_vel_w [T, B, 3].
  """
  dt = 1.0 / fps
  joint_vel = np.gradient(joint_pos, dt, axis=0)
  body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
  q = body_quat_w
  body_ang_vel_w = np.concatenate(
    [
      _angular_velocity(q[:1], q[1:2], dt),
      _angular_velocity(q[:-2], q[2:], 2.0 * dt),
      _angular_velocity(q[-2:-1], q[-1:], dt),
    ],
    axis=0,
  )
  return joint_vel, body_lin_vel_w, body_ang_vel_w


def clip_stats(name: str, motion: dict[str, np.ndarray], robot: str = "g1") -> dict:
  """Manifest statistics and feasibility verdict of a converted clip (npz arrays).

  The pelvis statistics use the robot's pelvis-equivalent (`tracked_bodies[0]`) and
  the height thresholds are scaled by its `height_scale`.
  """
  spec = get_robot(robot)
  fps = int(motion["fps"])
  z_min = motion["z_min"]
  pelvis = list(motion["body_names"]).index(spec.tracked_bodies[0])
  pelvis_quat = motion["body_quat_w"][:, pelvis].astype(np.float64)
  x, y = pelvis_quat[:, 1], pelvis_quat[:, 2]
  frames = len(z_min)
  return {
    "name": name,
    "frames": frames,
    "seconds": frames / fps,
    "zmin_min": float(z_min.min()),
    "zmin_max": float(z_min.max()),
    "feas_reject": feasibility_reject(
      z_min=z_min,
      foot_dist=motion["foot_dist"],
      pelvis_z=motion["body_pos_w"][:, pelvis, 2],
      pelvis_up_z=1.0 - 2.0 * (x * x + y * y),  # R[2, 2] of the pelvis rotation.
      body_speed=np.linalg.norm(motion["body_lin_vel_w"], axis=-1).mean(axis=1),
      fps=fps,
      height_scale=spec.height_scale,
    ),
  }


def convert_clip(
  csv_path: str,
  out_path: str,
  in_fps: int = SEED_FPS,
  out_fps: int = FPS,
  robot: str = "g1",
) -> dict:
  """Converts one SEED-format CSV of `robot` to an npz at out_path; returns its
  clip_stats."""
  spec = get_robot(robot)
  model = make_model(robot)
  root_pos, root_quat, joint_pos = resample(
    *parse_csv(csv_path, spec), in_fps, out_fps
  )
  fk = forward_kinematics(
    model,
    root_pos,
    root_quat,
    joint_pos,
    zmin_body_ids=zmin_bodies(model, spec.zmin_exclude_regex),
    feet=foot_geoms(model, spec.foot_geom_regex),
  )
  # Canonical quaternion sign (w >= 0).
  q = fk.body_quat_w
  body_quat_w = np.where(q[..., :1] < 0.0, -q, q)
  joint_vel, body_lin_vel_w, body_ang_vel_w = finite_diff_velocities(
    joint_pos, fk.body_pos_w, body_quat_w, out_fps
  )
  f32 = np.float32
  motion = {
    "fps": np.array(out_fps, dtype=np.int64),
    "joint_pos": joint_pos.astype(f32),
    "joint_vel": joint_vel.astype(f32),
    "body_pos_w": fk.body_pos_w.astype(f32),
    "body_quat_w": body_quat_w.astype(f32),
    "body_lin_vel_w": body_lin_vel_w.astype(f32),
    "body_ang_vel_w": body_ang_vel_w.astype(f32),
    "body_names": np.array(body_names(model)),
    "joint_names": np.array(spec.joint_names),
    "z_min": fk.z_min.astype(f32),
    "foot_dist": fk.foot_dist.astype(f32),
    "robot": np.array(spec.name),
    "root_body": np.array(spec.root_body),
  }
  # Write-then-rename so an interrupted run never leaves a truncated npz behind.
  out = Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.name + ".tmp")
  with open(tmp, "wb") as f:
    np.savez(f, **motion)
  os.replace(tmp, out)
  return clip_stats(Path(csv_path).stem, motion, robot)


def _convert_job(job: tuple[str, str, bool, str]) -> tuple[str, dict]:
  """Pool worker: converts (or re-reads an existing npz); returns (csv_path, stats)."""
  csv_path, out_path, overwrite, robot = job
  name = Path(csv_path).stem
  try:
    if overwrite or not os.path.exists(out_path):
      return csv_path, convert_clip(csv_path, out_path, robot=robot)
    with np.load(out_path) as z:
      return csv_path, clip_stats(name, {k: z[k] for k in z.files}, robot)
  # Recorded in the manifest: one bad file must not stop a 142k-clip run.
  except Exception as e:  # noqa: BLE001
    error = {"name": name, "frames": 0, "seconds": 0.0, "feas_reject": f"error: {e!r}"}
    return csv_path, error


def convert_all(
  csv_dir: str,
  out_dir: str,
  metadata_csv: str,
  workers: int,
  limit: int | None = None,
  overwrite: bool = False,
  robot: str = "g1",
) -> pd.DataFrame:
  """Converts `csv_dir/<session>/<clip>.csv` of `robot` to
  `out_dir/<session>/<clip>.npz`.

  Existing npz files are re-read for their stats instead of being reconverted unless
  `overwrite`. Applies the name filter, joins `category` / `package` from the SEED
  metadata (its `filename` column equals the CSV stem; clips without a metadata row
  get ""), writes `out_dir/manifest.csv` sorted by path and returns it. Clips that fail
  to convert get `feas_reject = "error: ..."`.
  """
  # An unknown robot or a broken model fails here, not once per clip in the workers.
  get_robot(robot)
  make_model(robot)
  csv_paths = sorted(Path(csv_dir).glob("*/*.csv"))[:limit]
  if not csv_paths:
    raise FileNotFoundError(f"no <session>/<clip>.csv files under {csv_dir}")
  rel_paths = [f"{p.parent.name}/{p.stem}.npz" for p in csv_paths]
  jobs = [
    (str(p), str(Path(out_dir) / r), overwrite, robot)
    for p, r in zip(csv_paths, rel_paths)
  ]
  with multiprocessing.Pool(workers) as pool:
    results = pool.imap_unordered(_convert_job, jobs, chunksize=8)
    stats = dict(tqdm(results, total=len(jobs), desc="convert"))

  rows = []
  for path, rel in zip(csv_paths, rel_paths):
    s = stats[str(path)]
    rows.append(
      {
        "name": path.stem,
        "path": rel,
        "session": path.parent.name,
        "frames": s["frames"],
        "seconds": s["seconds"],
        "name_reject": name_filter_reject(path.stem, path.parent.name) or "",
        "feas_reject": s["feas_reject"] or "",
      }
    )
  df = pd.DataFrame(rows)
  meta = pd.read_csv(metadata_csv, usecols=["filename", "category", "package"])
  meta = meta.drop_duplicates("filename").set_index("filename")
  for column in ("category", "package"):
    df[column] = df["name"].map(meta[column]).fillna("")
  df = df[list(MANIFEST_COLUMNS)]
  Path(out_dir).mkdir(parents=True, exist_ok=True)
  df.to_csv(Path(out_dir) / "manifest.csv", index=False)
  errors = df[df["feas_reject"].str.startswith("error")]
  if len(errors):
    first = errors.iloc[0]
    print(f"[WARNING] {len(errors)} clips failed, e.g. {first['path']}:")
    print(f"  {first['feas_reject']}")
  return df
