"""World-model state layout, canonical frames (heading, root) and canonical features
.

All functions take arbitrary leading batch dims, are differentiable and never modify
their inputs. Quaternions are wxyz.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_matrix,
  quat_inv,
  quat_mul,
  yaw_quat,
)
from torch import Tensor, nn

from morphmimic.constants import (
  ANCHOR_IDX,
  LOCAL_DIM,
  NUM_BODIES,
  NUM_JOINTS,
  ROOT_IDX,
  STATE_DIM,
)

# Section sizes of the 240-dim world-frame state, in layout order.
_STATE_SECTIONS = (
  NUM_JOINTS,
  NUM_JOINTS,
  NUM_BODIES * 3,
  NUM_BODIES * 4,
  NUM_BODIES * 3,
  NUM_BODIES * 3,
)
# Norm floor of the Gram-Schmidt normalisations in `quat_from_rot6d`.
_ROT6D_EPS = 1e-6

# Canonical frame of a state: the pelvis yaw (`heading_frame`) or the full pelvis
# orientation (`root_frame`).
FrameMode = Literal["heading", "root"]


class StateParts(NamedTuple):
  """World-frame state s split into named parts."""

  joint_pos: Tensor  # [..., 29]
  joint_vel: Tensor  # [..., 29]
  body_pos: Tensor  # [..., 14, 3]
  body_quat: Tensor  # [..., 14, 4]
  body_lin_vel: Tensor  # [..., 14, 3]
  body_ang_vel: Tensor  # [..., 14, 3]


class Frame(NamedTuple):
  """Canonical frame: origin at the pelvis xy (z = 0) and a unit orientation, the
  pelvis yaw (`heading_frame`) or the full pelvis orientation (`root_frame`)."""

  pos_xy: Tensor  # [..., 2]
  quat: Tensor  # [..., 4]


class LocalSlices:
  """Slices of the 268-dim canonical feature vector."""

  joint_pos = slice(0, 29)
  joint_vel = slice(29, 58)
  body_pos = slice(58, 100)  # 14 x 3
  body_rot6d = slice(100, 184)  # 14 x 6
  body_lin_vel = slice(184, 226)  # 14 x 3
  body_ang_vel = slice(226, 268)  # 14 x 3


def split_state(s: Tensor) -> StateParts:
  """[..., 240] -> StateParts (views into s)."""
  if s.shape[-1] != STATE_DIM:
    raise ValueError(f"Expected state last dim {STATE_DIM}, got {tuple(s.shape)}.")
  q, qd, pos, quat, lin, ang = torch.split(s, _STATE_SECTIONS, dim=-1)
  return StateParts(
    joint_pos=q,
    joint_vel=qd,
    body_pos=pos.unflatten(-1, (NUM_BODIES, 3)),
    body_quat=quat.unflatten(-1, (NUM_BODIES, 4)),
    body_lin_vel=lin.unflatten(-1, (NUM_BODIES, 3)),
    body_ang_vel=ang.unflatten(-1, (NUM_BODIES, 3)),
  )


def join_state(p: StateParts) -> Tensor:
  """StateParts -> [..., 240]."""
  return torch.cat(
    [
      p.joint_pos,
      p.joint_vel,
      p.body_pos.flatten(-2),
      p.body_quat.flatten(-2),
      p.body_lin_vel.flatten(-2),
      p.body_ang_vel.flatten(-2),
    ],
    dim=-1,
  )


def heading_frame(s: Tensor) -> Frame:
  """Heading frame of a state [..., 240] from the pelvis position and yaw. The yaw is
  undefined when the pelvis x-axis is vertical."""
  p = split_state(s)
  # mjlab's yaw_quat uses .view(-1, 4), which needs a contiguous input.
  root_quat = p.body_quat[..., ROOT_IDX, :].contiguous()
  return Frame(pos_xy=p.body_pos[..., ROOT_IDX, :2], quat=yaw_quat(root_quat))


def root_frame(s: Tensor) -> Frame:
  """Full pelvis frame of a state [..., 240]: the pelvis xy and the pelvis orientation
  (defined in every posture). The quaternion is renormalised, so a stored pelvis quat
  that is off unit norm (e.g. half-precision library storage) still gives a rotation."""
  p = split_state(s)
  quat = nn.functional.normalize(p.body_quat[..., ROOT_IDX, :], dim=-1)
  return Frame(pos_xy=p.body_pos[..., ROOT_IDX, :2], quat=quat)


def anchor_quat(s: Tensor) -> Tensor:
  """Full orientation [..., 4] (wxyz, world) of the anchor body (`ANCHOR_IDX`,
  torso_link) of a state [..., 240], renormalised as `root_frame`'s."""
  return nn.functional.normalize(split_state(s).body_quat[..., ANCHOR_IDX, :], dim=-1)


def state_frame(s: Tensor, mode: FrameMode) -> Frame:
  """Canonical frame of a state [..., 240]: `heading_frame` or `root_frame` by mode;
  ValueError for any other mode."""
  if mode == "heading":
    return heading_frame(s)
  if mode == "root":
    return root_frame(s)
  raise ValueError(f"Unknown frame mode {mode!r}; expected 'heading' or 'root'.")


def gravity_in_frame(frame: Frame) -> Tensor:
  """World gravity direction (0, 0, -1) expressed in the frame, [..., 3]; the constant
  (0, 0, -1) for a heading frame."""
  down = frame.quat.new_tensor((0.0, 0.0, -1.0)).expand(*frame.quat.shape[:-1], 3)
  return quat_apply_inverse(frame.quat, down)


def rot6d_from_quat(q: Tensor) -> Tensor:
  """[..., 4] -> [..., 6]: first two rotation-matrix columns, flattened row-major
  ([m00, m01, m10, m11, m20, m21]), the same layout as mjlab's tracking observations."""
  return matrix_from_quat(q)[..., :2].flatten(-2)


def quat_from_rot6d(r6: Tensor) -> Tensor:
  """[..., 6] -> [..., 4]. Gram-Schmidt orthonormalizes the two columns, so the input
  does not need to be orthonormal. The normalisations floor the norm at 1e-6 (not
  torch's 1e-12), so a near-degenerate column's gradient stays below ~1e6."""
  cols = r6.unflatten(-1, (3, 2))
  b1 = nn.functional.normalize(cols[..., 0], dim=-1, eps=_ROT6D_EPS)
  a2 = cols[..., 1]
  b2 = nn.functional.normalize(
    a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=_ROT6D_EPS
  )
  b3 = torch.linalg.cross(b1, b2, dim=-1)
  return quat_from_matrix(torch.stack([b1, b2, b3], dim=-1))


def _per_body(frame: Frame, batch_shape: torch.Size) -> tuple[Tensor, Tensor]:
  """Frame quat expanded to [..., 14, 4] and frame origin [..., 1, 3] (z = 0)."""
  rot = frame.quat.unsqueeze(-2).expand(*batch_shape, NUM_BODIES, 4)
  origin = nn.functional.pad(frame.pos_xy, (0, 1)).unsqueeze(-2)
  return rot, origin


def canonicalize(s: Tensor, frame: Frame) -> Tensor:
  """World-frame state [..., 240] -> features in the frame [..., 268] (LocalSlices).

  Positions are taken relative to the frame origin (the pelvis xy at z = 0); positions,
  rotations and velocities are rotated by the inverse frame rotation; joints are
  unchanged. In a heading frame (yaw only) local z is the world height.
  """
  p = split_state(s)
  rot, origin = _per_body(frame, s.shape[:-1])
  pos = quat_apply_inverse(rot, p.body_pos - origin)
  rot6d = rot6d_from_quat(quat_mul(quat_inv(rot), p.body_quat))
  lin = quat_apply_inverse(rot, p.body_lin_vel)
  ang = quat_apply_inverse(rot, p.body_ang_vel)
  return torch.cat(
    [
      p.joint_pos,
      p.joint_vel,
      pos.flatten(-2),
      rot6d.flatten(-2),
      lin.flatten(-2),
      ang.flatten(-2),
    ],
    dim=-1,
  )


def decanonicalize(local: Tensor, frame: Frame) -> Tensor:
  """Features in the frame [..., 268] -> world-frame state [..., 240]; inverts
  `canonicalize` for the same frame."""
  if local.shape[-1] != LOCAL_DIM:
    raise ValueError(f"Expected local last dim {LOCAL_DIM}, got {tuple(local.shape)}.")
  rot, origin = _per_body(frame, local.shape[:-1])
  pos = local[..., LocalSlices.body_pos].unflatten(-1, (NUM_BODIES, 3))
  rot6d = local[..., LocalSlices.body_rot6d].unflatten(-1, (NUM_BODIES, 6))
  lin = local[..., LocalSlices.body_lin_vel].unflatten(-1, (NUM_BODIES, 3))
  ang = local[..., LocalSlices.body_ang_vel].unflatten(-1, (NUM_BODIES, 3))
  return join_state(
    StateParts(
      joint_pos=local[..., LocalSlices.joint_pos],
      joint_vel=local[..., LocalSlices.joint_vel],
      body_pos=quat_apply(rot, pos) + origin,
      body_quat=quat_mul(rot, quat_from_rot6d(rot6d)),
      body_lin_vel=quat_apply(rot, lin),
      body_ang_vel=quat_apply(rot, ang),
    )
  )


class StateNormalizer(nn.Module):
  """Fixed per-dim affine normalizer for canonical features (buffers, not params)."""

  mean: Tensor
  std: Tensor

  def __init__(self, dim: int = LOCAL_DIM) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("std", torch.ones(dim))

  @classmethod
  def from_stats(
    cls, mean: Tensor, std: Tensor, floor: float = 1e-2
  ) -> StateNormalizer:
    """Builds a normalizer from per-dim stats; std is clamped from below at `floor`."""
    if mean.ndim != 1 or mean.shape != std.shape:
      raise ValueError(
        f"Expected matching 1-D mean/std, got {tuple(mean.shape)} / {tuple(std.shape)}."
      )
    norm = cls(mean.shape[0])
    norm.mean = mean.detach().clone()
    norm.std = std.detach().clamp(min=floor)
    return norm

  def normalize(self, x: Tensor) -> Tensor:
    return (x - self.mean) / self.std

  def denormalize(self, x: Tensor) -> Tensor:
    return x * self.std + self.mean
