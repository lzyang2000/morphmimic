"""Human skeleton goal: features, relative orientation, target jitter and masks
(keypoint goal version 10).

The goal is the operator's 24 SMPL joints (`HUMAN_JOINTS`) over the K future frames,
each a world pose. Positions and orientations are in the heading frame of the anchor
at frame 0 (positions with the origin at its xy, z absolute): the Pelvis slot when it
is on, else the Head. The only robot input is the human Spine3's orientation relative
to the robot anchor body's (torso_link) full orientation at each future frame,
R_k = inv(robot torso) * spine3_k * inv(rest), rest = Rz(`HUMAN_YAW_OFFSET`) *
Rx(`HUMAN_ROLL_OFFSET`), as rot6d after the mask bits (the "heading" block, K x 6); the
identity when the robot torso is upright and faces the way the human chest faces. It
is the human counterpart of the motion goal's orientation term (the reference torso
in the robot torso frame), the term the motion teacher's waist correction reads. A
row without the Spine3 slot (five and three points) measures its position anchor
instead (Pelvis, else Head; `heading_slot`), still against the robot torso, so every
setting sees the robot's waist.
(Version 3 measured all 24 orientations against the robot pelvis, so a small robot
roll made every target look rolled the other way; version 4 had no robot input and
could not correct heading drift; version 5 gave the yaw error alone, so a robot lean
left the goal unchanged; versions 6-7 measured the human anchor against the robot
pelvis, so a robot waist lean left the goal unchanged (the teacher given only that
loses its waist the same way).) The block is
standardised by its own normaliser like the features; then `mask_human_goal` zeroes
masked slots and frames in the features and the block's rows whose heading mask is
off and frames whose frame mask is off. All functions take arbitrary leading batch
dims except where noted. Quaternions are wxyz.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_angle_axis,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  yaw_quat,
)
from torch import Tensor, nn

from morphmimic.constants import (
  GOAL_K,
  HUMAN_DRIFT,
  HUMAN_FEAT_DIM,
  HUMAN_FIVE_IDX,
  HUMAN_HEAD,
  HUMAN_HEADING_DIM,
  HUMAN_PELVIS,
  HUMAN_POINT_DIM,
  HUMAN_POSE_DIM,
  HUMAN_RAW_DIM,
  HUMAN_ROLL_OFFSET,
  HUMAN_SIX_IDX,
  HUMAN_SPINE3,
  HUMAN_THREE_IDX,
  HUMAN_YAW_OFFSET,
  NUM_HUMAN,
)
from morphmimic.model.state import rot6d_from_quat

HumanMaskMode = Literal["mixed", "full", "five", "six", "three"]
HumanPoints = Literal["full", "five", "six", "three"]

_POINT_IDX: dict[str, tuple[int, ...]] = {
  "five": HUMAN_FIVE_IDX,
  "six": HUMAN_SIX_IDX,
  "three": HUMAN_THREE_IDX,
}

# `mixed` masks: all slots with probability _P_FULL, else the five-point set; the
# horizon is the full preview, else none, else uniform.
_P_FULL = 0.5
_P_FULL_PREVIEW = 0.5
_P_NO_PREVIEW = 0.2


def pack_human_goal(pos_w: Tensor, quat_w: Tensor, robot_quat: Tensor) -> Tensor:
  """Raw goal from the human world poses `pos_w` [..., K, 24, 3], `quat_w`
  [..., K, 24, 4] and the robot anchor body's (torso_link) quaternion `robot_quat`
  [..., 4] (wxyz, world; `state.anchor_quat`) -> [..., 1684]: frame by frame, slot by slot, [pos 3 | quat 4], then the
  robot quaternion."""
  poses = torch.cat([pos_w, quat_w], dim=-1).flatten(-3)
  return torch.cat([poses, robot_quat], dim=-1)


def split_human_goal(raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """The inverse of `pack_human_goal`: [..., 1684] -> (pos_w, quat_w, robot_quat)."""
  poses = raw[..., :HUMAN_RAW_DIM].unflatten(-1, (GOAL_K, NUM_HUMAN, HUMAN_POSE_DIM))
  return poses[..., :3], poses[..., 3:], raw[..., HUMAN_RAW_DIM:]


def anchor_slot(point_mask: Tensor) -> Tensor:
  """Anchor slot per row from slot masks [..., 24] -> [...] long: the Pelvis when its
  slot is on, else the Head."""
  return torch.where(point_mask[..., HUMAN_PELVIS], HUMAN_PELVIS, HUMAN_HEAD)


def heading_slot(point_mask: Tensor) -> Tensor:
  """Slot of the heading block per row from slot masks [..., 24] -> [...] long: the
  Spine3 when its slot is on, else the position anchor (`anchor_slot`)."""
  spine3 = point_mask[..., HUMAN_SPINE3]
  return torch.where(spine3, HUMAN_SPINE3, anchor_slot(point_mask))


def heading_features(robot_quat: Tensor, quat_w: Tensor, point_mask: Tensor) -> Tensor:
  """The human Spine3's (else the anchor's) orientation relative to the robot torso,
  rot6d per future frame [..., K * 6], unmasked (`mask_human_goal` masks it after the
  normalisation).

  R_k = inv(`robot_quat` [..., 4]) * slot_k * Rx(-`HUMAN_ROLL_OFFSET`) *
  Rz(-`HUMAN_YAW_OFFSET`), with `robot_quat` the robot anchor body's (torso_link)
  orientation and slot_k the quaternion at frame k of `quat_w` [..., K, 24, 4] of the
  row's `heading_slot` (`point_mask` [..., 24]): the identity when the robot torso is
  upright and faces the way the human chest (pelvis, head) faces. The SMPL rest frame
  is the robot pelvis frame turned by Rz(`HUMAN_YAW_OFFSET`) * Rx(`HUMAN_ROLL_OFFSET`);
  on the retargeted references the Spine3-to-torso_link offset is within 3° of the same
  rotation (1 h subset), and the block's normaliser takes out the constant rest.
  """
  slot = heading_slot(point_mask)
  idx = slot[..., None, None, None].expand(*slot.shape, GOAL_K, 1, 4)
  anchor = torch.gather(quat_w, -2, idx).squeeze(-2)
  roll, yaw = HUMAN_ROLL_OFFSET / 2, HUMAN_YAW_OFFSET / 2
  undo = quat_mul(
    anchor.new_tensor([[math.cos(roll), -math.sin(roll), 0.0, 0.0]]),
    anchor.new_tensor([[math.cos(yaw), 0.0, 0.0, -math.sin(yaw)]]),
  )
  robot = quat_inv(robot_quat)[..., None, :].expand_as(anchor)
  rel = quat_mul(robot, quat_mul(anchor, undo.expand_as(anchor)))
  return rot6d_from_quat(rel).flatten(-2)


def heading_from_goal(raw: Tensor, point_mask: Tensor) -> Tensor:
  """`heading_features` from a raw goal [..., 1684] (`pack_human_goal`)."""
  _, quat_w, robot_quat = split_human_goal(raw)
  return heading_features(robot_quat, quat_w, point_mask)


def human_features(
  pos_w: Tensor,
  quat_w: Tensor,
  point_mask: Tensor,
  jitter: Tensor | None = None,
) -> Tensor:
  """Goal features [..., 2160] from the human world poses `pos_w` [..., K, 24, 3],
  `quat_w` [..., K, 24, 4].

  Per frame and slot: [position | rot6d orientation]. `point_mask` [..., 24] picks the
  anchor per row: slot 0 (Pelvis) if on, else slot 15 (Head). Both are in the
  anchor's heading frame at frame 0: positions with the origin at its xy, z absolute;
  orientations `quat_inv(anchor_yaw) * quat`. The mask does not zero anything here.

  `jitter` [..., K, 24, 6] is target noise: (dx, dy, dz) is added to the position and
  (droll, dpitch, dyaw) is composed onto the orientation as a body-local rotation.
  """
  anchor = anchor_slot(point_mask)
  idx = anchor[..., None, None]
  anchor_pos = torch.gather(pos_w[..., 0, :, :], -2, idx.expand(*anchor.shape, 1, 3))
  anchor_quat = torch.gather(quat_w[..., 0, :, :], -2, idx.expand(*anchor.shape, 1, 4))
  anchor_yaw = yaw_quat(anchor_quat.squeeze(-2).contiguous())
  anchor_yaw = anchor_yaw[..., None, None, :].expand(*pos_w.shape[:-1], 4)
  origin = nn.functional.pad(anchor_pos[..., 0, :2], (0, 1))[..., None, None, :]
  pos = quat_apply_inverse(anchor_yaw, pos_w - origin)
  quat = quat_mul(quat_inv(anchor_yaw), quat_w)
  if jitter is not None:
    pos = pos + jitter[..., :3]
    quat = quat_mul(
      quat, quat_from_euler_xyz(jitter[..., 3], jitter[..., 4], jitter[..., 5])
    )
  return torch.cat([pos, rot6d_from_quat(quat)], dim=-1).flatten(-3)


def human_features_from_goal(
  raw: Tensor, point_mask: Tensor, jitter: Tensor | None = None
) -> Tensor:
  """`human_features` from a raw goal [..., 1684] (`pack_human_goal`)."""
  pos_w, quat_w, _ = split_human_goal(raw)
  return human_features(pos_w, quat_w, point_mask, jitter)


def human_jitter(
  shape: tuple[int, ...], half: dict[str, float], device: torch.device
) -> Tensor:
  """Uniform target jitter [*shape, 24, 6] in [-h, h] per dim, `half` as
  `HUMAN_JITTER`: h = pos_xy for dims 0, 1; pos_z for 2; rot_rp for 3, 4; rot_yaw
  for 5."""
  h = torch.tensor(
    [
      half["pos_xy"],
      half["pos_xy"],
      half["pos_z"],
      half["rot_rp"],
      half["rot_rp"],
      half["rot_yaw"],
    ],
    device=device,
  )
  return (2.0 * torch.rand(*shape, NUM_HUMAN, 6, device=device) - 1.0) * h


def sample_human_drift(batch: int, device: torch.device) -> Tensor:
  """Per-row drift parameters [B, 15] (`HUMAN_DRIFT`): xy direction (2), rotation
  axis (3), xy amplitude, z offset, z amplitude, rotation amplitude, then the angular
  frequencies (3) and phases (3) of the xy, z and rotation sinusoids."""
  d = HUMAN_DRIFT
  lo, hi = math.log(d["freq"][0]), math.log(d["freq"][1])
  omega = 2.0 * math.pi * torch.exp(lo + (hi - lo) * torch.rand(batch, 3, device=device))
  u = torch.rand(batch, 4, device=device)
  return torch.cat(
    [
      nn.functional.normalize(torch.randn(batch, 2, device=device), dim=-1),
      nn.functional.normalize(torch.randn(batch, 3, device=device), dim=-1),
      (u[:, 0] * d["xy_speed"] / omega[:, 0])[:, None],  # peak speed / omega
      ((2.0 * u[:, 1] - 1.0) * d["z_offset"])[:, None],
      (u[:, 2] * d["z_amp"])[:, None],
      (u[:, 3] * d["rot_amp"])[:, None],
      omega,
      2.0 * math.pi * torch.rand(batch, 3, device=device),
    ],
    dim=-1,
  )


def drift_human_poses(
  pos_w: Tensor, quat_w: Tensor, drift: Tensor, time_s: Tensor
) -> tuple[Tensor, Tensor]:
  """The human world poses `pos_w` [B, T, 24, 3], `quat_w` [B, T, 24, 4] moved as one
  rigid body by the drift `drift` [B, 15] (`sample_human_drift`) at the times
  `time_s` [B or 1, T] (s), plus fresh per-frame noise (`HUMAN_DRIFT`): turned about
  the frame's Pelvis, then shifted."""
  b, t = pos_w.shape[:2]
  device = pos_w.device
  wave = torch.sin(drift[:, None, 9:12] * time_s[..., None] + drift[:, None, 12:15])
  xy = drift[:, None, :2] * (drift[:, None, 5:6] * wave[..., 0:1])
  z = drift[:, None, 6:7] + drift[:, None, 7:8] * wave[..., 1:2]
  radius = HUMAN_DRIFT["pos_noise"] * torch.rand(b, t, 1, device=device)
  offset = torch.cat([xy, z], dim=-1) + radius * nn.functional.normalize(
    torch.randn(b, t, 3, device=device), dim=-1
  )
  turn = quat_from_angle_axis(
    drift[:, None, 8] * wave[..., 2], drift[:, None, 2:5].expand(b, t, 3)
  )
  noise = quat_from_angle_axis(
    HUMAN_DRIFT["rot_noise"] * (2.0 * torch.rand(b, t, device=device) - 1.0),
    nn.functional.normalize(torch.randn(b, t, 3, device=device), dim=-1),
  )
  rot = quat_mul(noise, turn)[..., None, :].expand(*quat_w.shape)
  pelvis = pos_w[..., HUMAN_PELVIS : HUMAN_PELVIS + 1, :]
  pos = pelvis + quat_apply(rot, pos_w - pelvis) + offset[..., None, :]
  return pos, quat_mul(rot, quat_w)


def drift_human_goal(raw: Tensor, drift: Tensor, time_s: Tensor) -> Tensor:
  """`drift_human_poses` on the human poses of a raw goal [B, 1684]
  (`pack_human_goal`) at the frame times `time_s` [B, K]; the robot quaternion stays."""
  pos_w, quat_w, robot_quat = split_human_goal(raw)
  pos, quat = drift_human_poses(pos_w, quat_w, drift, time_s)
  return pack_human_goal(pos, quat, robot_quat)


def _prefix_frame_mask(horizon: Tensor) -> Tensor:
  """Visible preview lengths [B] -> prefix frame masks [B, K]: frame k is visible iff
  k < horizon."""
  return torch.arange(GOAL_K, device=horizon.device) < horizon[:, None]


def _with_anchor(points: Tensor) -> Tensor:
  """Slot masks [B, 24] with the Pelvis slot turned on in every row that has neither
  the Pelvis nor the Head on, so each row has a position anchor."""
  out = points.clone()
  out[..., HUMAN_PELVIS] |= ~(points[..., HUMAN_PELVIS] | points[..., HUMAN_HEAD])
  return out


def fixed_points(batch: int, points: HumanPoints, device: torch.device) -> Tensor:
  """Slot masks [B, 24] of one setting: `full` (all on), `five`, `six` or `three`."""
  if points == "full":
    return torch.ones(batch, NUM_HUMAN, dtype=torch.bool, device=device)
  if points not in _POINT_IDX:
    raise ValueError(
      f"Expected points 'full', 'five', 'six' or 'three', got {points!r}."
    )
  mask = torch.zeros(NUM_HUMAN, dtype=torch.bool, device=device)
  mask[list(_POINT_IDX[points])] = True
  return mask.expand(batch, NUM_HUMAN).clone()


def sample_human_masks(
  batch: int,
  mode: HumanMaskMode,
  device: torch.device,
  heading_frac: float = 1.0,
  preview_mix: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
  """Training masks: slot mask [B, 24], prefix frame mask [B, K] and heading mask
  [B], all bool.

  `full` / `five` / `six` / `three`: that setting and h = K. `mixed`: all slots with
  probability 0.5, else the five-point set; the visible horizon h is K with
  probability 0.5, 1 with 0.2, else uniform in {1, ..., K} when `preview_mix` is
  True, and always K when it is False (the robot runs 0.2 s behind the operator at
  deploy, so the headset always supplies the full preview). Every row has the Pelvis
  or the Head on. In every mode the heading is on with probability `heading_frac`.
  """
  if not 0.0 <= heading_frac <= 1.0:
    raise ValueError(f"Expected 0 <= heading_frac <= 1, got {heading_frac}.")
  heading = torch.rand(batch, device=device) < heading_frac
  if mode in ("full", "five", "six", "three"):
    points = fixed_points(batch, mode, device)
    horizon = torch.full((batch,), GOAL_K, device=device)
    return _with_anchor(points), _prefix_frame_mask(horizon), heading
  if mode != "mixed":
    raise ValueError(
      f"Expected mode 'mixed', 'full', 'five', 'six' or 'three', got {mode!r}."
    )
  full = torch.rand(batch, 1, device=device) < _P_FULL
  points = torch.where(
    full, fixed_points(batch, "full", device), fixed_points(batch, "five", device)
  )
  uniform = torch.randint(1, GOAL_K + 1, (batch,), device=device)
  if not preview_mix:
    horizon = torch.full_like(uniform, GOAL_K)
  else:
    v = torch.rand(batch, device=device)
    horizon = torch.where(
      v < _P_FULL_PREVIEW,
      torch.full_like(uniform, GOAL_K),
      torch.where(
        v < _P_FULL_PREVIEW + _P_NO_PREVIEW, torch.ones_like(uniform), uniform
      ),
    )
  return _with_anchor(points), _prefix_frame_mask(horizon), heading


def fixed_human_masks(
  batch: int,
  points: HumanPoints,
  horizon: int,
  device: torch.device,
  heading: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
  """Evaluation masks: the `points` setting, a visible horizon of `horizon` frames
  (1 <= horizon <= K) and the heading on or off, repeated over the batch."""
  if not 1 <= horizon <= GOAL_K:
    raise ValueError(f"Expected 1 <= horizon <= {GOAL_K}, got {horizon}.")
  frames = _prefix_frame_mask(torch.full((batch,), horizon, device=device))
  heading_mask = torch.full((batch,), heading, dtype=torch.bool, device=device)
  return fixed_points(batch, points, device), frames, heading_mask


def mask_human_goal(
  feats_norm: Tensor,
  point_mask: Tensor,
  frame_mask: Tensor,
  heading_norm: Tensor,
  heading_mask: Tensor,
) -> Tensor:
  """Normalised features [B, 2160] and normalised heading block [B, K * 6] -> masked
  goal [B, 2254].

  Every (frame k, slot p) block is zeroed when `point_mask[p]` or `frame_mask[k]` is
  False; the mask bits [point_mask (24), frame_mask (K)] are appended as floats, then
  `heading_norm` (`heading_features` through its normaliser) with a row zeroed where
  `heading_mask` [B] is False and frame k's 6 numbers zeroed where `frame_mask[k]` is
  False, so a hidden frame shows nothing. The zeros are exact (`torch.where`), whatever
  the normalised values are.
  """
  if feats_norm.shape[-1] != HUMAN_FEAT_DIM:
    raise ValueError(
      f"Expected features last dim {HUMAN_FEAT_DIM}, got {tuple(feats_norm.shape)}."
    )
  if heading_norm.shape[-1] != HUMAN_HEADING_DIM:
    raise ValueError(
      f"Expected heading last dim {HUMAN_HEADING_DIM},"
      f" got {tuple(heading_norm.shape)}."
    )
  blocks = feats_norm.unflatten(-1, (GOAL_K, NUM_HUMAN, HUMAN_POINT_DIM))
  visible = (frame_mask[..., :, None] & point_mask[..., None, :])[..., None]
  masked = torch.where(visible, blocks, torch.zeros_like(blocks))
  rel = heading_norm.to(feats_norm.dtype).unflatten(-1, (GOAL_K, 6))
  shown = (frame_mask & heading_mask[..., None])[..., None]
  rel_shown = torch.where(shown, rel, torch.zeros_like(rel))
  return torch.cat(
    [
      masked.flatten(-3),
      point_mask.to(feats_norm.dtype),
      frame_mask.to(feats_norm.dtype),
      rel_shown.flatten(-2),
    ],
    dim=-1,
  )
