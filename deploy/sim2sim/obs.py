"""Policy inputs from what a robot measures, in numpy only (no torch, no mjlab).

The robot side of the sim2sim (and later the hardware) loop builds the inputs of
`policy.onnx` (`deploy/export_onnx.py`) from joint encoders, the IMU (the orientation
and gyro of the body it sits on, `Robot.imu_body`: the G1 and X2 pelvis, the T2 trunk)
and the reference motion:

- `proprio_frame`: one 93-value frame, as the env's `policy` group (`env_cfg.
  _proprio_terms`): canonical joint positions minus the default pose, canonical joint
  velocities, the gyro and the gravity direction in the IMU frame, the last action.
- `goal_motion`: the 670-value motion goal, `rewards_torch.goal_features` (goal
  version 2): reference joints, the reference anchor path in its own heading frame,
  the reference anchor orientation relative to the robot anchor. Under the "robot"
  goal anchor (`--goal-anchor robot`) the anchor path is in the robot anchor's frame
  instead (goal version 1's), which needs the robot's position.

The goal needs the orientation of the anchor (torso), and the references turn to the
heading of slot 0 (the pelvis-equivalent); `Robot.anchor_quat` and
`Robot.slot0_quat` compute them from the IMU and the joint encoders through the robot's
kinematic tree. Under the "robot" anchor, `Robot.anchor_pose` adds the anchor's
position from the floating base's position and yaw (a pose source). Each function
mirrors its torch original. Quaternions are wxyz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_JOINTS = 29
NUM_BODIES = 14
GOAL_K = 10
ROOT_IDX = 0
ANCHOR_IDX = 7
PROPRIO_DIM = 93
STATE_DIM = 240
ROBOTS_DIR = Path(__file__).parent / "robots"

# World-model state layout [q 29 | qd 29 | pos 14x3 | quat 14x4 | lin 14x3 | ang 14x3].
_Q = slice(0, 29)
_QD = slice(29, 58)
_POS = slice(58, 100)
_QUAT = slice(100, 156)


# Quaternions (wxyz), broadcasting over leading dims.


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    axis=-1,
  )


def quat_inv(q: np.ndarray) -> np.ndarray:
  """Inverse (conjugate over the squared norm, as mjlab's `quat_inv`)."""
  conj = q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)
  return conj / np.sum(q * q, axis=-1, keepdims=True).clip(min=1e-9)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotates `v` by the unit quaternion `q`."""
  w = q[..., :1]
  xyz = q[..., 1:]
  t = 2.0 * np.cross(xyz, v)
  return v + w * t + np.cross(xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotates `v` by the inverse of the unit quaternion `q`."""
  w = q[..., :1]
  xyz = q[..., 1:]
  t = 2.0 * np.cross(xyz, v)
  return v - w * t + np.cross(xyz, t)


def yaw_quat(q: np.ndarray) -> np.ndarray:
  """The yaw part of `q` (mjlab's `yaw_quat`)."""
  w, x, y, z = np.moveaxis(q, -1, 0)
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  zero = np.zeros_like(yaw)
  return np.stack((np.cos(yaw / 2.0), zero, zero, np.sin(yaw / 2.0)), axis=-1)


def slerp(qa: np.ndarray, qb: np.ndarray, alpha: float) -> np.ndarray:
  """Unit quaternions `qa` -> `qb` [..., 4] by `alpha`, along the shortest arc."""
  dot = np.sum(qa * qb, axis=-1, keepdims=True)
  qb = np.where(dot < 0.0, -qb, qb)
  theta = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))
  sin = np.sin(theta)
  near = sin < 1e-8
  safe = np.where(near, 1.0, sin)
  wa = np.where(near, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / safe)
  wb = np.where(near, alpha, np.sin(alpha * theta) / safe)
  q = wa * qa + wb * qb
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def axis_angle_quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
  half = np.asarray(angle)[..., None] / 2.0
  return np.concatenate((np.cos(half), np.sin(half) * axis), axis=-1)


def rot6d(q: np.ndarray) -> np.ndarray:
  """First two rotation-matrix columns, row-major (`state.rot6d_from_quat`; mjlab's
  `matrix_from_quat`, which divides by the squared norm)."""
  r, i, j, k = np.moveaxis(q, -1, 0)
  s = 2.0 / np.sum(q * q, axis=-1)
  return np.stack(
    (
      1 - s * (j * j + k * k), s * (i * j - k * r),
      s * (i * j + k * r), 1 - s * (i * i + k * k),
      s * (i * k - j * r), s * (j * k + i * r),
    ),
    axis=-1,
  )


# The robot: joint map, default pose, action scale, kinematic tree.

BODIES: dict[str, str] = {"g1_amber": "g1"}
"""Robots that are another robot's body with other actuator constants: the body's
neutral stand, GMR retargeting and hardware node serve them."""


@dataclass(frozen=True)
class Robot:
  """A robot's constants, baked from the training code by
  `deploy/sim2sim/export_assets.py` into `robots/<name>.json`."""

  name: str
  native_joint_names: tuple[str, ...]
  default_joint_pos: np.ndarray
  """Native default pose [J]."""
  action_scale: np.ndarray
  """Native action scale [J]."""
  perm: np.ndarray
  """Native index of each canonical slot [29]: q_canon = sign * (q[perm] - offset)."""
  sign: np.ndarray
  offset: np.ndarray
  tracked_bodies: tuple[str, ...]
  imu_body: str
  """The body the IMU sits on; its orientation and gyro are what the robot reports."""
  fk_parent: tuple[int, ...]
  """Kinematic tree from the root body: each body's parent (-1 for the root)."""
  fk_quat: np.ndarray
  """Each tree body's orientation in its parent at zero joint angles [n, 4]."""
  fk_joints: tuple[tuple[tuple[int, np.ndarray, float], ...], ...]
  """Each tree body's hinge joints: (native joint index, axis, reference angle)."""
  fk_pos: np.ndarray | None
  """Each tree body's position in its parent [n, 3] (the root's is zero); None in
  constants baked before it (`anchor_pose` then raises)."""
  fk_slot0: int
  fk_anchor: int
  fk_imu: int
  physics_dt: float
  decimation: int
  kp: np.ndarray | None = None
  """Native PD stiffness [J]."""
  kd: np.ndarray | None = None
  effort_limit: np.ndarray | None = None
  dc_saturation: np.ndarray | None = None
  """DC-motor joints' stall torque [J], 0 for the built-in PD joints; None if none is."""
  dc_velocity_limit: np.ndarray | None = None
  """DC-motor joints' no-load speed [J] (the torque-speed curve reaches 0 there)."""
  dc_law: str = "mjlab"
  """The DC motors' torque-speed law: "mjlab" (full braking torque at any speed) or
  "booster" (the bound falls with |qd| in both directions; the T2)."""

  @property
  def num_native(self) -> int:
    return len(self.native_joint_names)

  @property
  def control_dt(self) -> float:
    return self.physics_dt * self.decimation

  @property
  def body(self) -> str:
    """The robot whose body this is (`BODIES`): the key of `stand.json`."""
    return BODIES.get(self.name, self.name)

  @classmethod
  def load(cls, name: str, robots_dir: Path = ROBOTS_DIR) -> Robot:
    """The baked constants `robots_dir/<name>.json`."""
    return cls.from_dict(json.loads((Path(robots_dir) / f"{name}.json").read_text()))

  @classmethod
  def from_dict(cls, d: dict) -> Robot:
    """From a constants dict: a baked `robots/<name>.json`, or the `robot` entry of a
    deployment folder's `<name>.json` (`deploy/export_onnx.py`)."""
    tr, fk, dc = d["translator"], d["fk"], d.get("dc_motor")
    return cls(
      name=d["robot"],
      native_joint_names=tuple(d["native_joint_names"]),
      default_joint_pos=np.asarray(d["default_joint_pos"], np.float64),
      action_scale=np.asarray(d["action_scale"], np.float64),
      perm=np.asarray(tr["perm"], np.int64),
      sign=np.asarray(tr["sign"], np.float64),
      offset=np.asarray(tr["offset"], np.float64),
      tracked_bodies=tuple(d["tracked_bodies"]),
      imu_body=d["imu_body"],
      fk_parent=tuple(b["parent"] for b in fk["bodies"]),
      fk_quat=np.asarray([b["quat"] for b in fk["bodies"]], np.float64),
      fk_joints=tuple(
        tuple((j["native_index"], np.asarray(j["axis"], np.float64), j["ref"])
              for j in b["joints"])
        for b in fk["bodies"]
      ),
      fk_pos=(np.asarray([b["pos"] for b in fk["bodies"]], np.float64)
              if all("pos" in b for b in fk["bodies"]) else None),
      fk_slot0=fk["slot0"],
      fk_anchor=fk["anchor"],
      fk_imu=fk["imu"],
      physics_dt=d["timing"]["physics_dt"],
      decimation=d["timing"]["decimation"],
      kp=np.asarray(d["pd"]["kp"], np.float64),
      kd=np.asarray(d["pd"]["kd"], np.float64),
      effort_limit=np.asarray(d["pd"]["effort_limit"], np.float64),
      dc_saturation=None if dc is None else np.asarray(dc["saturation_effort"], np.float64),
      dc_velocity_limit=None if dc is None else np.asarray(dc["velocity_limit"], np.float64),
      dc_law="mjlab" if dc is None else dc.get("law", "mjlab"),
    )

  def to_canonical(self, q: np.ndarray) -> np.ndarray:
    return self.sign * (q[..., self.perm] - self.offset)

  def to_canonical_delta(self, x: np.ndarray) -> np.ndarray:
    return self.sign * x[..., self.perm]

  def targets(self, action: np.ndarray) -> np.ndarray:
    """Canonical action [29] -> native joint targets [J] (`CanonicalJointPositionAction`:
    default + to_native_delta(a * scale); untracked joints held at the default)."""
    delta = np.zeros(self.num_native)
    delta[self.perm] = self.sign * action * self.action_scale[self.perm]
    return self.default_joint_pos + delta

  def _tree_quats(self, q: np.ndarray) -> np.ndarray:
    """Orientation of every tree body in the root body's frame [n, 4]."""
    out = np.zeros((len(self.fk_parent), 4))
    for b, parent in enumerate(self.fk_parent):
      rot = np.array([1.0, 0.0, 0.0, 0.0]) if parent < 0 else quat_mul(
        out[parent], self.fk_quat[b])
      for joint, axis, ref in self.fk_joints[b]:
        rot = quat_mul(rot, axis_angle_quat(axis, q[joint] - ref))
      out[b] = rot
    return out

  def _from_imu(self, imu_quat: np.ndarray, q: np.ndarray, body: int) -> np.ndarray:
    """World orientation of tree body `body` from the IMU body's world orientation and
    the native joint positions."""
    rel = self._tree_quats(q)
    root = quat_mul(imu_quat, quat_inv(rel[self.fk_imu]))
    return quat_mul(root, rel[body])

  def anchor_quat(self, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """World orientation of the anchor body (torso)."""
    return self._from_imu(imu_quat, q, self.fk_anchor)

  def slot0_quat(self, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """World orientation of tracked slot 0 (the pelvis-equivalent)."""
    return self._from_imu(imu_quat, q, self.fk_slot0)

  def anchor_pose(self, base_pos: np.ndarray, base_yaw: float, imu_quat: np.ndarray,
                  q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World position [3] and orientation [4] of the anchor body (torso) from the
    floating base's (the tree's root body: the G1 and X2 pelvis, the T2 trunk) world
    position and yaw (rad), as a pose source gives them, the IMU orientation (the
    base's roll and pitch) and the native joint positions. The yaw is the pose
    source's, so position and orientation share its world frame."""
    if self.fk_pos is None:
      raise ValueError(f"{self.name}: the robot constants have no body offsets (`fk` "
                       "`pos`); re-export them (export_assets.py --bake, export_onnx.py)")
    rel = self._tree_quats(q)
    root = quat_mul(imu_quat, quat_inv(rel[self.fk_imu]))
    turn = axis_angle_quat(np.array([0.0, 0.0, 1.0]), np.asarray(base_yaw))
    root = quat_mul(turn, quat_mul(quat_inv(yaw_quat(root)), root))
    pos = np.zeros((len(self.fk_parent), 3))
    for b, parent in enumerate(self.fk_parent):
      if parent >= 0:
        pos[b] = pos[parent] + quat_apply(rel[parent], self.fk_pos[b])
    anchor = self.fk_anchor
    return (np.asarray(base_pos, np.float64) + quat_apply(root, pos[anchor]),
            quat_mul(root, rel[anchor]))


# Policy inputs.


def proprio_frame(robot: Robot, q: np.ndarray, dq: np.ndarray, imu_quat: np.ndarray,
                  gyro: np.ndarray, last_action: np.ndarray) -> np.ndarray:
  """One proprioception frame [93] (`env_cfg._proprio_terms`, no encoder bias): the
  IMU's orientation `imu_quat` gives the gravity direction, `gyro` is its angular
  velocity in its own frame, as the IMU reports it."""
  gravity = quat_apply_inverse(imu_quat, np.array([0.0, 0.0, -1.0]))
  return np.concatenate((
    robot.to_canonical(q) - robot.to_canonical(robot.default_joint_pos),
    robot.to_canonical_delta(dq),
    gyro,
    gravity,
    last_action,
  )).astype(np.float32)


class History:
  """The last `length` proprioception frames, oldest first; the first frame after a
  reset fills every slot (`ObsHistory.reset`)."""

  def __init__(self, length: int) -> None:
    self.length = length
    self.frames: np.ndarray | None = None

  def reset(self) -> None:
    self.frames = None

  def push(self, frame: np.ndarray) -> np.ndarray:
    if self.frames is None:
      self.frames = np.repeat(frame[None], self.length, axis=0)
    else:
      self.frames = np.concatenate((self.frames[1:], frame[None]), axis=0)
    return self.frames.reshape(-1)


def _bodies(states: np.ndarray, part: slice, width: int) -> np.ndarray:
  return states[..., part].reshape(*states.shape[:-1], NUM_BODIES, width)


def goal_motion(ref: np.ndarray, robot_anchor_quat: np.ndarray,
                robot_anchor_pos: np.ndarray | None = None,
                anchor: str = "reference") -> np.ndarray:
  """Motion goal [K * 67] from the K reference states `ref` [K, 240] and the robot
  anchor's world orientation (and, under `anchor="robot"`, its world position, in the
  reference's world) (`rewards_torch.goal_features(..., anchor)`)."""
  pos, quat = _bodies(ref, _POS, 3), _bodies(ref, _QUAT, 4)
  if anchor == "robot":
    if robot_anchor_pos is None:
      raise ValueError("the robot goal anchor needs the robot anchor position")
    anchor_pos = quat_apply_inverse(robot_anchor_quat[None],
                                    pos[:, ANCHOR_IDX] - robot_anchor_pos[None])
  elif anchor == "reference":
    ref_yaw = yaw_quat(quat[0, ROOT_IDX])
    origin = np.array([pos[0, ROOT_IDX, 0], pos[0, ROOT_IDX, 1], 0.0])
    anchor_pos = quat_apply_inverse(ref_yaw[None], pos[:, ANCHOR_IDX] - origin)
  else:
    raise ValueError(f"Expected anchor 'reference' or 'robot', got {anchor!r}.")
  anchor_rel = quat_mul(quat_inv(robot_anchor_quat)[None], quat[:, ANCHOR_IDX])
  frames = np.concatenate((ref[:, _Q], ref[:, _QD], anchor_pos, rot6d(anchor_rel)), axis=-1)
  return frames.reshape(-1).astype(np.float32)


def reference_window(states: np.ndarray, frame: int, k: int = GOAL_K) -> np.ndarray:
  """Reference states of frames frame .. frame + k - 1, held at the clip's last frame
  (`MotionLibrary.future`)."""
  idx = np.minimum(np.arange(frame, frame + k), len(states) - 1)
  return states[idx]
