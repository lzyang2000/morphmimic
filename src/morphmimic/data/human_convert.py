"""SOMA BVH -> 50 Hz human sidecar of the 24 SMPL slots.

A sidecar holds `human_pos_w [T, 24, 3]` (m, z-up), `human_quat_w [T, 24, 4]` (wxyz,
w >= 0), `human_joint_names` (`HUMAN_JOINTS`), `fps` (50) and `source_frames` (the
length of the 120 Hz source). It lives at
`<dataset root>/human50/<date>/<stem>.npz` beside the robot clip `<root>/<date>/<stem>.npz`
and has the robot clip's frames: both are resampled with the same rule.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from morphmimic.constants import (
  FPS,
  HUMAN_JOINTS,
  HUMAN_L_ANKLE,
  HUMAN_PELVIS,
  HUMAN_R_ANKLE,
  HUMAN_SOMA_SOURCE,
  ROOT_IDX,
)
from morphmimic.data.bvh import load_bvh_poses

SOMA_FPS = 120
# +90 degrees about x (wxyz): Y-up to z-up, (x, y, z) -> (x, -z, y).
_QX90 = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
# World orientations (wxyz, Y-up, as `load_bvh_poses` gives them) of the 24 slots'
# SOMA joints in the SOMA T-pose: frame 0 of the SOMA base rig
# `soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh` (`soma_rest_quats`; the rig
# is a test fixture). SOMA's joint frames are bone-aligned (the bone runs along local
# +x), so they are removed to get SMPL's rest convention (identity at the T-pose).
SOMA_REST_QUAT = np.array([
  [0.494476, 0.503983, 0.495985, 0.505464],  # Pelvis <- Hips
  [0.499996, -0.500004, 0.499998, -0.500002],  # Left_Hip <- LeftLeg
  [0.500004, 0.499996, 0.500001, 0.499999],  # Right_Hip <- RightLeg
  [0.493460, 0.505930, 0.494371, 0.506091],  # Spine1 <- Spine1
  [0.500002, -0.499998, 0.500003, -0.499997],  # Left_Knee <- LeftShin
  [0.499997, 0.500003, 0.499997, 0.500003],  # Right_Knee <- RightShin
  [0.540019, 0.460320, 0.536289, 0.457034],  # Spine2 <- Spine2
  [0.125462, -0.695887, 0.138475, -0.693415],  # Left_Ankle <- LeftFoot
  [0.695887, 0.125462, 0.693415, 0.138475],  # Right_Ankle <- RightFoot
  [0.508690, 0.493859, 0.487270, 0.509810],  # Spine3 <- Chest
  [0.104964, -0.690212, 0.094309, -0.709715],  # Left_Foot <- LeftToeBase
  [0.696178, 0.103439, 0.703710, 0.097109],  # Right_Foot <- RightToeBase
  [0.401366, 0.580059, 0.408852, 0.579031],  # Neck <- Neck1
  [0.695285, 0.695285, 0.128761, -0.128761],  # Left_Collar <- LeftShoulder
  [0.695285, -0.695285, -0.128761, -0.128761],  # Right_Collar <- RightShoulder
  [0.527405, 0.471554, 0.528046, 0.469732],  # Head <- Head
  [0.707103, 0.707110, 0.000031, -0.000031],  # Left_Shoulder <- LeftArm
  [0.707108, -0.707106, -0.000033, -0.000033],  # Right_Shoulder <- RightArm
  [0.707106, 0.707107, -0.000033, 0.000033],  # Left_Elbow <- LeftForeArm
  [0.707107, -0.707107, 0.000034, 0.000035],  # Right_Elbow <- RightForeArm
  [0.658595, 0.752325, -0.000758, 0.016093],  # Left_Wrist <- LeftHand
  [0.752325, -0.658595, 0.016093, 0.000758],  # Right_Wrist <- RightHand
  [0.682753, 0.725113, 0.038811, -0.080949],  # Left_Hand <- LeftHandMiddle1
  [0.720783, -0.687680, -0.078726, -0.037022],  # Right_Hand <- RightHandMiddle1
]
)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
  return q[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
  return q[..., [1, 2, 3, 0]]


def YUP_CM_TO_ZUP_M(
  pos: np.ndarray, quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Y-up centimetres to z-up metres: positions [..., 3] / 100 then
  (x, y, z) -> (x, -z, y); orientations [..., 4] wxyz left-multiplied by that
  rotation (+90 degrees about x)."""
  p = pos / 100.0
  p = np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)
  w0, x0 = _QX90[0], _QX90[1]
  w, x, y, z = np.moveaxis(quat, -1, 0)
  q = np.stack([w0 * w - x0 * x, w0 * x + x0 * w, w0 * y - x0 * z, w0 * z + x0 * y], -1)
  return p, q


def resample_poses(
  pos: np.ndarray, quat: np.ndarray, in_fps: int, out_fps: int
) -> tuple[np.ndarray, np.ndarray]:
  """[T, J, 3], [T, J, 4] wxyz -> [T', J, 3], [T', J, 4] at t_k = k / out_fps,
  k = 0..floor(duration * out_fps), as `seed_convert.resample`: positions linearly
  interpolated, each joint's quaternions slerped."""
  n_in = len(pos)
  duration = (n_in - 1) / in_fps
  n_out = int(np.floor(duration * out_fps + 1e-9)) + 1
  t_out = np.minimum(np.arange(n_out) / out_fps, duration)
  x = t_out * in_fps
  i0 = np.clip(np.floor(x).astype(np.int64), 0, max(n_in - 2, 0))
  i1 = np.minimum(i0 + 1, n_in - 1)
  w = (x - i0)[:, None, None]
  out_pos = pos[i0] * (1.0 - w) + pos[i1] * w
  t_in = np.arange(n_in) / in_fps
  out_quat = np.empty((n_out, quat.shape[1], 4))
  for j in range(quat.shape[1]):
    slerp = Slerp(t_in, Rotation.from_quat(_wxyz_to_xyzw(quat[:, j])))
    out_quat[:, j] = _xyzw_to_wxyz(slerp(t_out).as_quat())
  return out_pos, out_quat


def soma_rest_quats(base_rig_bvh: Path) -> np.ndarray:
  """[24, 4] wxyz world orientations (Y-up) of the slots' SOMA joints at frame 0 of the
  base rig BVH (its T-pose), in `HUMAN_SOMA_SOURCE` order."""
  _, quat, names, _ = load_bvh_poses(Path(base_rig_bvh))
  return quat[0, [names[n] for n in HUMAN_SOMA_SOURCE]]


def soma_slot_poses(
  bvh_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, float]:
  """The 24 slots of a SOMA BVH at its own frame rate: positions [T, 24, 3] (m, z-up),
  orientations [T, 24, 4] (wxyz, z-up) in SMPL's rest convention, i.e. each joint's
  SOMA rest frame removed (`q * inv(SOMA_REST_QUAT[j])`, Y-up) before the axis change,
  and the frame time (s). At the T-pose every slot's orientation is the axis change
  itself."""
  pos, quat, names, frame_time = load_bvh_poses(Path(bvh_path))
  for name in HUMAN_SOMA_SOURCE:
    if name not in names:
      raise ValueError(f"{bvh_path}: missing SOMA joint {name!r}")
  idx = [names[n] for n in HUMAN_SOMA_SOURCE]
  rest_inv = Rotation.from_quat(_wxyz_to_xyzw(SOMA_REST_QUAT)).inv().as_quat()
  slots = Rotation.from_quat(_wxyz_to_xyzw(quat[:, idx]).reshape(-1, 4))
  corrected = slots * Rotation.from_quat(np.tile(rest_inv, (len(quat), 1)))
  quat = _xyzw_to_wxyz(corrected.as_quat().reshape(len(quat), len(idx), 4))
  pos, quat = YUP_CM_TO_ZUP_M(pos[:, idx], quat)
  return pos, quat, frame_time


def convert_human_clip(
  bvh_path: str, out_path: str, in_fps: int = SOMA_FPS, out_fps: int = FPS
) -> dict:
  """Converts one SOMA BVH to a human sidecar at out_path (`soma_slot_poses`, then
  resampled; keeps the BVH's own root path); returns {name, frames, seconds,
  source_frames, feet_min_z}. The BVH's frame time must match `in_fps`."""
  pos, quat, frame_time = soma_slot_poses(bvh_path)
  if round(1.0 / frame_time) != in_fps:
    raise ValueError(f"{bvh_path}: frame time {frame_time} s is not {in_fps} fps")
  source_frames = len(pos)
  pos, quat = resample_poses(pos, quat, in_fps, out_fps)
  # Canonical quaternion sign (w >= 0).
  quat = np.where(quat[..., :1] < 0.0, -quat, quat)
  f32 = np.float32
  sidecar = {
    "human_pos_w": pos.astype(f32),
    "human_quat_w": quat.astype(f32),
    "human_joint_names": np.array(HUMAN_JOINTS),
    "fps": np.array(out_fps, dtype=np.int64),
    "source_frames": np.array(source_frames, dtype=np.int64),
  }
  # Write-then-rename so an interrupted run never leaves a truncated npz behind.
  out = Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.name + ".tmp")
  with open(tmp, "wb") as f:
    np.savez(f, **sidecar)
  os.replace(tmp, out)
  return {
    "name": Path(bvh_path).stem,
    "frames": len(pos),
    "seconds": (len(pos) - 1) / out_fps,
    "source_frames": source_frames,
    "feet_min_z": _feet_min_z(sidecar["human_pos_w"]),
  }


def _feet_min_z(pos: np.ndarray) -> float:
  """Lowest z over frames of the lower of the two ankle slots."""
  return float(pos[:, [HUMAN_L_ANKLE, HUMAN_R_ANKLE], 2].min())


def sidecar_path(clip_path: str | Path, human_root: str | Path | None = None) -> Path:
  """`<human_root or <clip root>/human50>/<date>/<stem>.npz` of `<root>/<date>/<stem>.npz`."""
  clip = Path(clip_path)
  root = Path(human_root) if human_root is not None else clip.parent.parent / "human50"
  return root / clip.parent.name / clip.name


def bvh_path(clip_path: str | Path, bvh_root: str | Path) -> Path:
  """`<bvh_root>/<date>/<stem>.bvh` of the robot clip `<root>/<date>/<stem>.npz`."""
  clip = Path(clip_path)
  return Path(bvh_root) / clip.parent.name / (clip.stem + ".bvh")


def clip_pelvis_body(z: Mapping[str, np.ndarray], body_names: list[str]) -> str:
  """The pelvis-equivalent body of a robot clip `z` (an open npz with `body_names`):
  the tracked pelvis slot (`ROOT_IDX`) of the registry robot the npz names (`robot`:
  the G1 and X2 `pelvis`, the T2 `waist_yaw_link`), else its `root_body`, else its
  first body (the floating base; the G1 clips name neither and start with `pelvis`)."""
  from morphmimic.robots import ROBOTS, VARIANTS, get_robot

  if "robot" in z and str(z["robot"]) in (*ROBOTS, *VARIANTS):
    return get_robot(str(z["robot"])).tracked_bodies[ROOT_IDX]
  if "root_body" in z:
    return str(z["root_body"])
  return body_names[0]


def check_clip(clip_path: str, human_root: str | Path | None = None) -> dict:
  """Compares a robot clip with its sidecar: frame counts, the feet's lowest z,
  `height_ratio` (robot over human start pelvis z: the retarget's scale) and the RMS
  xy distance (m) of the human Pelvis path, scaled by `height_ratio`, from the robot
  pelvis path, both shifted to start at the origin (over the shorter of the two). The
  robot pelvis is `clip_pelvis_body`'s (the T2's is its `waist_yaw_link`). A missing
  sidecar counts as 0 human frames, with `sidecar` False."""
  name = Path(clip_path).stem
  with np.load(clip_path) as z:
    robot_frames = len(z["joint_pos"])
    body_names = [str(n) for n in z["body_names"]]
    pelvis = clip_pelvis_body(z, body_names)
    robot = z["body_pos_w"][:, body_names.index(pelvis)].astype(np.float64)
  side = sidecar_path(clip_path, human_root)
  if not side.exists():
    return {
      "name": name, "robot_frames": robot_frames, "human_frames": 0,
      "frames_match": False, "feet_min_z": float("nan"), "height_ratio": float("nan"),
      "pelvis_xy_rms_m": float("nan"), "sidecar": False,
    }
  with np.load(side) as h:
    human_pos = h["human_pos_w"].astype(np.float64)
  n = min(robot_frames, len(human_pos))
  human = human_pos[:n, HUMAN_PELVIS]
  ratio = robot[0, 2] / human[0, 2]
  d = (human[:, :2] - human[0, :2]) * ratio - (robot[:n, :2] - robot[0, :2])
  return {
    "name": name,
    "robot_frames": robot_frames,
    "human_frames": len(human_pos),
    "frames_match": robot_frames == len(human_pos),
    "feet_min_z": _feet_min_z(human_pos),
    "height_ratio": float(ratio),
    "pelvis_xy_rms_m": float(np.sqrt((d**2).sum(-1).mean())),
    "sidecar": True,
  }
