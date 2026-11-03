"""Analytic world-model heads: tracking reward, termination features, observation and
goal.

Each function mirrors, on plain tensors, the mjlab BeyondMimic tracking term it
replaces, so that the simulator and imagination see the same quantities. Inputs are
world-frame states [..., 240] (see `morphmimic.model.state`); all functions take
arbitrary leading batch dims, are differentiable and never modify their inputs.
Quaternions are wxyz. Anchor = torso_link (tracked body 7), root = pelvis (body 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
  yaw_quat,
)
from torch import Tensor, nn

from morphmimic.constants import (
  ANCHOR_IDX,
  CRITIC_ANCHOR_POS_SCALE,
  DT,
  GOAL_ANCHORS,
  NUM_BODIES,
  ROOT_IDX,
  GoalAnchor,
)
from morphmimic.model.state import heading_frame, rot6d_from_quat, split_state


@dataclass(frozen=True)
class RewardCfg:
  """Tracking reward weights and kernel stds (paper Table A.2, mjlab G1 tracking)."""

  w_body_pos: float = 1.0
  w_body_ori: float = 1.0
  w_body_lin_vel: float = 1.0
  w_body_ang_vel: float = 1.0
  w_root_pos: float = 0.5
  w_root_ori: float = 0.5
  w_action_rate: float = -0.1
  w_joint_limit: float = -2.0
  std_pos: float = 0.3
  std_ori: float = 0.4
  std_lin_vel: float = 1.0
  std_ang_vel: float = 3.14
  dt: float = DT


_DEFAULT_REWARD_CFG = RewardCfg()


class ProprioSlices(NamedTuple):
  """Slices of the 93-dim analytic (proprioceptive) observation."""

  joint_pos: slice
  joint_vel: slice
  ang_vel: slice
  gravity: slice
  prev_action: slice


PROPRIO_SLICES = ProprioSlices(
  joint_pos=slice(0, 29),  # q + encoder bias - q0
  joint_vel=slice(29, 58),
  ang_vel=slice(58, 61),  # pelvis angular velocity, pelvis frame
  gravity=slice(61, 64),  # gravity direction, pelvis frame
  prev_action=slice(64, 93),
)


class GoalFrameSlices(NamedTuple):
  """Slices of one 67-dim goal frame."""

  joint_pos: slice
  joint_vel: slice
  anchor_pos: slice
  anchor_rot6d: slice


goal_frame_slices = GoalFrameSlices(
  joint_pos=slice(0, 29),  # reference q
  joint_vel=slice(29, 58),  # reference qd
  anchor_pos=slice(58, 61),  # reference anchor position, reference heading frame
  anchor_rot6d=slice(61, 67),  # reference anchor rotation, robot anchor frame
)


def _per_body(x: Tensor) -> Tensor:
  """[..., D] -> [..., 14, D] (mjlab's quat ops need identical shapes)."""
  return x.unsqueeze(-2).expand(*x.shape[:-1], NUM_BODIES, x.shape[-1])


def _gravity_like(x: Tensor) -> Tensor:
  """Gravity direction (0, 0, -1) with the shape, dtype and device of x [..., 3].

  Built on x's device (no host-to-device copy, so no sync per call).
  """
  return torch.cat([torch.zeros_like(x[..., :2]), -torch.ones_like(x[..., 2:])], dim=-1)


def _quat_error_sq(q1: Tensor, q2: Tensor) -> Tensor:
  """`quat_error_magnitude(q1, q2) ** 2` for unit quaternions [..., 4] -> [...].

  Same value as mjlab's version (rotation angle of q1 q2^-1, in [0, pi]), but its
  gradient stays finite at zero error, where mjlab's Taylor-branch `torch.where`
  backpropagates 0/0 = NaN.
  """
  diff = quat_mul(q1, quat_conjugate(q2))
  sin_half = torch.linalg.vector_norm(diff[..., 1:], dim=-1)
  angle = 2.0 * torch.atan2(sin_half, diff[..., 0].abs())
  return angle.square()


def relative_reference(state: Tensor, ref: Tensor) -> tuple[Tensor, Tensor]:
  """Reference body poses re-expressed at the robot anchor's xy and heading.

  Mirrors mjlab's `MotionCommand.update_relative_body_poses`: the reference is moved
  so that its anchor sits at (robot anchor x, robot anchor y, reference anchor z) and
  rotated by the yaw of robot_anchor_quat * ref_anchor_quat^-1.

  Returns body positions [..., 14, 3] and body quaternions [..., 14, 4].
  """
  robot, motion = split_state(state), split_state(ref)
  robot_anchor_pos = robot.body_pos[..., ANCHOR_IDX, :]
  robot_anchor_quat = robot.body_quat[..., ANCHOR_IDX, :]
  ref_anchor_pos = motion.body_pos[..., ANCHOR_IDX, :]
  ref_anchor_quat = motion.body_quat[..., ANCHOR_IDX, :]

  delta_pos = torch.cat([robot_anchor_pos[..., :2], ref_anchor_pos[..., 2:]], dim=-1)
  delta_ori = _per_body(
    yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
  )
  body_quat = quat_mul(delta_ori, motion.body_quat)
  body_pos = delta_pos.unsqueeze(-2) + quat_apply(
    delta_ori, motion.body_pos - ref_anchor_pos.unsqueeze(-2)
  )
  return body_pos, body_quat


def tracking_reward(
  state: Tensor,
  ref: Tensor,
  action: Tensor,
  prev_action: Tensor,
  joint_limits: Tensor,
  cfg: RewardCfg = _DEFAULT_REWARD_CFG,
) -> tuple[Tensor, dict[str, Tensor]]:
  """Per-step tracking reward of `state` [..., 240] against reference `ref` [..., 240].

  Terms follow mjlab's tracking rewards (global anchor pose, anchor-relative body
  poses, world-frame body velocities, all body terms averaged over the 14 bodies), plus
  the action-rate and joint-limit penalties. `action`/`prev_action` are raw policy
  actions [..., 29]; `joint_limits` [29, 2] are the (soft) joint (min, max) limits.

  Returns (total [...], terms) where each term is already multiplied by its weight and
  dt, and total is their sum. Perfect tracking inside the limits gives 5 * dt.
  """
  robot, motion = split_state(state), split_state(ref)
  body_pos_rel, body_quat_rel = relative_reference(state, ref)

  root_pos_err = torch.sum(
    torch.square(
      motion.body_pos[..., ANCHOR_IDX, :] - robot.body_pos[..., ANCHOR_IDX, :]
    ),
    dim=-1,
  )
  root_ori_err = _quat_error_sq(
    motion.body_quat[..., ANCHOR_IDX, :], robot.body_quat[..., ANCHOR_IDX, :]
  )
  body_pos_err = torch.sum(torch.square(body_pos_rel - robot.body_pos), dim=-1)
  body_ori_err = _quat_error_sq(body_quat_rel, robot.body_quat)
  body_lin_vel_err = torch.sum(
    torch.square(motion.body_lin_vel - robot.body_lin_vel), dim=-1
  )
  body_ang_vel_err = torch.sum(
    torch.square(motion.body_ang_vel - robot.body_ang_vel), dim=-1
  )

  q = robot.joint_pos
  q_min, q_max = joint_limits[..., 0], joint_limits[..., 1]
  out_of_limits = -(q - q_min).clip(max=0.0) + (q - q_max).clip(min=0.0)

  weight_and_value = {
    "root_pos": (cfg.w_root_pos, torch.exp(-root_pos_err / cfg.std_pos**2)),
    "root_ori": (cfg.w_root_ori, torch.exp(-root_ori_err / cfg.std_ori**2)),
    "body_pos": (cfg.w_body_pos, torch.exp(-body_pos_err.mean(-1) / cfg.std_pos**2)),
    "body_ori": (cfg.w_body_ori, torch.exp(-body_ori_err.mean(-1) / cfg.std_ori**2)),
    "body_lin_vel": (
      cfg.w_body_lin_vel,
      torch.exp(-body_lin_vel_err.mean(-1) / cfg.std_lin_vel**2),
    ),
    "body_ang_vel": (
      cfg.w_body_ang_vel,
      torch.exp(-body_ang_vel_err.mean(-1) / cfg.std_ang_vel**2),
    ),
    "action_rate": (
      cfg.w_action_rate,
      torch.sum(torch.square(action - prev_action), dim=-1),
    ),
    "joint_limit": (cfg.w_joint_limit, torch.sum(out_of_limits, dim=-1)),
  }
  terms = {
    name: value * weight * cfg.dt for name, (weight, value) in weight_and_value.items()
  }
  total = torch.stack(list(terms.values())).sum(dim=0)
  return total, terms


def termination_features(state: Tensor, ref: Tensor) -> tuple[Tensor, Tensor]:
  """Anchor deviations behind mjlab's `bad_anchor_pos_z_only` / `bad_anchor_ori`.

  Returns e_z = |ref anchor z - robot anchor z| and e_tilt = |g_ref,z - g_robot,z|,
  where g is the gravity direction (0, 0, -1) in the respective anchor frame; both
  [...].
  """
  robot, motion = split_state(state), split_state(ref)
  robot_anchor_pos = robot.body_pos[..., ANCHOR_IDX, :]
  e_z = torch.abs(motion.body_pos[..., ANCHOR_IDX, 2] - robot_anchor_pos[..., 2])
  gravity = _gravity_like(robot_anchor_pos)
  g_ref = quat_apply_inverse(motion.body_quat[..., ANCHOR_IDX, :], gravity)
  g_robot = quat_apply_inverse(robot.body_quat[..., ANCHOR_IDX, :], gravity)
  e_tilt = torch.abs(g_ref[..., 2] - g_robot[..., 2])
  return e_z, e_tilt


def hard_termination(
  state: Tensor, ref: Tensor, z_thresh: float = 0.25, tilt_thresh: float = 0.8
) -> Tensor:
  """Simulator termination (time-outs excluded): e_z > z_thresh or e_tilt >
  tilt_thresh. Bool [...]."""
  e_z, e_tilt = termination_features(state, ref)
  return (e_z > z_thresh) | (e_tilt > tilt_thresh)


def soft_termination(
  state: Tensor,
  ref: Tensor,
  z_thresh: float = 0.25,
  tilt_thresh: float = 0.8,
  z_scale: float = 0.02,
  tilt_scale: float = 0.05,
) -> Tensor:
  """Differentiable termination probability [...]:
  1 - (1 - sigmoid((e_z - z_thresh) / z_scale)) (1 - sigmoid((e_tilt - tilt_thresh) /
  tilt_scale))."""
  e_z, e_tilt = termination_features(state, ref)
  p_z = torch.sigmoid((e_z - z_thresh) / z_scale)
  p_tilt = torch.sigmoid((e_tilt - tilt_thresh) / tilt_scale)
  return 1.0 - (1.0 - p_z) * (1.0 - p_tilt)


def analytic_obs(
  state: Tensor,
  encoder_bias: Tensor,
  default_joint_pos: Tensor,
  prev_action: Tensor,
  imu_idx: int = ROOT_IDX,
) -> Tensor:
  """Noise-free policy observation [..., 93] (PROPRIO_SLICES):
  [q + encoder_bias - q0 | qd | IMU body angular velocity in its frame (gyro) |
  gravity direction in the IMU body frame | prev_action]. `imu_idx` is the tracked
  slot of the IMU body (`RobotSpec.imu_slot`: 0 for the G1 and X2 pelvis, 7 for the
  T2 trunk)."""
  p = split_state(state)
  imu_quat = p.body_quat[..., imu_idx, :]
  imu_ang_vel = p.body_ang_vel[..., imu_idx, :]
  ang_vel_b = quat_apply_inverse(imu_quat, imu_ang_vel)
  gravity_b = quat_apply_inverse(imu_quat, _gravity_like(imu_ang_vel))
  return torch.cat(
    [
      p.joint_pos + encoder_bias - default_joint_pos,
      p.joint_vel,
      ang_vel_b,
      gravity_b,
      prev_action,
    ],
    dim=-1,
  )


def goal_features(
  state: Tensor, ref_future: Tensor, anchor: GoalAnchor = "reference"
) -> Tensor:
  """Goal from K future reference frames `ref_future` [..., K, 240] -> [..., K * 67].

  Per frame (goal_frame_slices): [ref q | ref qd | ref anchor position in the
  reference's own heading frame | ref anchor rot6d orientation in the current robot
  anchor frame]. Frames are concatenated in time order.

  Position-free: the robot's position never enters. The anchor
  position is taken in `heading_frame(ref_future[..., 0, :])`, the reference pelvis
  heading frame of the first goal frame (origin at its pelvis xy, z absolute), so it
  gives the reference's path ahead and its anchor height. The orientation is relative
  to the robot anchor, as mjlab's `motion_anchor_ori_b`: on hardware it comes from
  the IMU orientation (yaw included) and the waist joints.

  `anchor="robot"` (`--goal-anchor robot`, the motion interface only) gives goal
  version 1's features instead: the reference anchor position and orientation in the
  current robot anchor frame (mjlab's `motion_anchor_pos_b` / `motion_anchor_ori_b`),
  so the robot's position enters; at deploy it needs a pose source (state estimator).
  The orientation term is the same under both anchors.
  """
  if anchor not in GOAL_ANCHORS:
    raise ValueError(f"Expected anchor 'reference' or 'robot', got {anchor!r}.")
  robot, motion = split_state(state), split_state(ref_future)
  frame_shape = ref_future.shape[:-1]
  robot_anchor_quat = robot.body_quat[..., None, ANCHOR_IDX, :].expand(*frame_shape, 4)
  if anchor == "robot":
    robot_anchor_pos = robot.body_pos[..., None, ANCHOR_IDX, :].expand(*frame_shape, 3)
    anchor_pos, anchor_quat_b = subtract_frame_transforms(
      robot_anchor_pos,
      robot_anchor_quat,
      motion.body_pos[..., ANCHOR_IDX, :],
      motion.body_quat[..., ANCHOR_IDX, :],
    )
  else:
    ref_frame = heading_frame(ref_future[..., 0, :])
    ref_yaw = ref_frame.quat[..., None, :].expand(*frame_shape, 4)
    origin = nn.functional.pad(ref_frame.pos_xy, (0, 1))[..., None, :]
    anchor_pos = quat_apply_inverse(
      ref_yaw, motion.body_pos[..., ANCHOR_IDX, :] - origin
    )
    anchor_quat_b = quat_mul(
      quat_inv(robot_anchor_quat), motion.body_quat[..., ANCHOR_IDX, :]
    )
  frames = torch.cat(
    [
      motion.joint_pos,
      motion.joint_vel,
      anchor_pos,
      rot6d_from_quat(anchor_quat_b),
    ],
    dim=-1,
  )
  return frames.flatten(-2)


def critic_anchor_pos(state: Tensor, ref_future: Tensor) -> Tensor:
  """Privileged critic input from K future reference frames `ref_future` [..., K, 240]
  -> [..., K * 3]: each frame's reference anchor position in the current robot anchor
  frame (mjlab's `motion_anchor_pos_b`, the goal-version-1 anchor position), divided
  by `CRITIC_ANCHOR_POS_SCALE`.

  The critics only: the reward's global anchor-position term
  depends on the robot's offset from the reference, which the position-free goal
  (`goal_features`) does not carry.
  """
  robot, motion = split_state(state), split_state(ref_future)
  frame_shape = ref_future.shape[:-1]
  robot_anchor_pos = robot.body_pos[..., None, ANCHOR_IDX, :].expand(*frame_shape, 3)
  robot_anchor_quat = robot.body_quat[..., None, ANCHOR_IDX, :].expand(*frame_shape, 4)
  anchor_pos_b, _ = subtract_frame_transforms(
    robot_anchor_pos, robot_anchor_quat, motion.body_pos[..., ANCHOR_IDX, :]
  )
  return (anchor_pos_b / CRITIC_ANCHOR_POS_SCALE).flatten(-2)
