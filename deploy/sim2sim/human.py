"""The human skeleton goal at deploy (keypoint goal version 10), numpy only.

The policy node's copy of `morphmimic.model.human`'s feature build (`human_features`,
`heading_features`; a test checks they agree) and the window operations its human
references need (`turn_human`, `blend_human`, `human_heading`). A human frame is the 24
SMPL slots' world poses `[24 * 7]`, per slot the position (m, z-up), then the
orientation (wxyz) in SMPL's rest convention: one frame of a training sidecar
(`human_pos_w`, `human_quat_w`) or of the human streamer (`vr_teleop.human_target`). A
window is K frames `[K, 168]`, frame 0 the current one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from deploy.sim2sim.obs import (
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  rot6d,
  slerp,
  yaw_quat,
)

# PICO's 24 body joints (XRoboToolkit, SMPL order; the goal's slots, `HUMAN_JOINTS` in
# `morphmimic.constants`) and their SMPL parents.
PICO_JOINTS = (
  "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee", "Spine2",
  "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot", "Neck",
  "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
  "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
)  # fmt: skip
SMPL_PARENT = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18,
               19, 20, 21)  # fmt: skip
NUM_HUMAN = 24
HUMAN_FRAME_DIM = NUM_HUMAN * 7
HUMAN_PELVIS, HUMAN_SPINE3, HUMAN_HEAD = 0, 9, 15
# The slots of each setting but "full" (all 24): `morphmimic.constants` HUMAN_*_IDX.
HUMAN_POINTS = {"five": (0, 7, 8, 20, 21), "six": (0, 9, 7, 8, 20, 21), "three": (15, 20, 21)}
# `morphmimic.constants`: an aligned human joint is the robot body turned by
# Rz(HUMAN_YAW_OFFSET) then Rx(HUMAN_ROLL_OFFSET), so a human orientation times _UNDO is
# in the robot's convention (x forward, z up).
HUMAN_YAW_OFFSET = math.pi / 2
HUMAN_ROLL_OFFSET = math.pi / 2
_UNDO = quat_mul(
  np.array([math.cos(HUMAN_ROLL_OFFSET / 2), -math.sin(HUMAN_ROLL_OFFSET / 2), 0.0, 0.0]),
  np.array([math.cos(HUMAN_YAW_OFFSET / 2), 0.0, 0.0, -math.sin(HUMAN_YAW_OFFSET / 2)]),
)
# The uniform skeleton's bones and its neutral stand (`vr_teleop.human_target`).
SKELETON_FILE = Path(__file__).parents[1] / "vr_teleop" / "human_skeleton.json"


def point_mask(points: str) -> np.ndarray:
  """Slot mask [24] bool of a setting: "full", "five", "six" or "three"."""
  if points == "full":
    return np.ones(NUM_HUMAN, bool)
  mask = np.zeros(NUM_HUMAN, bool)
  mask[list(HUMAN_POINTS[points])] = True
  return mask


def split_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Human frames [..., 168] -> positions [..., 24, 3] and orientations [..., 24, 4]."""
  slots = frames.reshape(*frames.shape[:-1], NUM_HUMAN, 7)
  return slots[..., :3], slots[..., 3:]


def join_frames(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
  """The inverse of `split_frames`."""
  return np.concatenate([pos, quat], -1).reshape(*pos.shape[:-2], HUMAN_FRAME_DIM)


def _anchor(mask: np.ndarray) -> int:
  return HUMAN_PELVIS if mask[HUMAN_PELVIS] else HUMAN_HEAD


def human_features(window: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """The goal features [K * 24 * 9] of a window [K, 168] (`model.human.human_features`):
  per frame and slot the position and the rot6d orientation, in the heading frame of the
  anchor at frame 0 (the Pelvis when `mask` [24] has it, else the Head), positions with
  the origin at its xy, z absolute. Unmasked: the ONNX graph masks after normalising."""
  pos, quat = split_frames(window)
  a = _anchor(mask)
  yaw = yaw_quat(quat[0, a])
  pos = quat_apply_inverse(yaw, pos - np.array([pos[0, a, 0], pos[0, a, 1], 0.0]))
  return np.concatenate([pos, rot6d(quat_mul(quat_inv(yaw), quat))], -1).ravel()


def heading_features(robot_quat: np.ndarray, window: np.ndarray, mask: np.ndarray
                     ) -> np.ndarray:
  """The heading block [K * 6] (`model.human.heading_features`): per frame the human
  Spine3's orientation (else the anchor's) relative to the robot torso `robot_quat`
  [4] (wxyz, world), inv(robot) * slot * _UNDO, as rot6d; the identity when the torso
  is upright and faces the way the human chest faces."""
  _, quat = split_frames(window)
  slot = HUMAN_SPINE3 if mask[HUMAN_SPINE3] else _anchor(mask)
  return rot6d(quat_mul(quat_inv(robot_quat), quat_mul(quat[:, slot], _UNDO))).ravel()


def human_heading(frame: np.ndarray) -> float:
  """Heading (rad) of a human frame [168]: the yaw of its pelvis in the robot's
  convention, comparable with the robot pelvis's."""
  w, x, y, z = quat_mul(split_frames(frame)[1][HUMAN_PELVIS], _UNDO)
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def turn_human(frames: np.ndarray, yaw: float, origin: np.ndarray) -> np.ndarray:
  """Human frames [..., 168] turned by `yaw` about the vertical through `origin` (x, y),
  which goes to x = y = 0 (as `teleop.turn_states`)."""
  pos, quat = split_frames(frames)
  turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  pos = quat_apply(turn, pos - np.array([origin[0], origin[1], 0.0]))
  return join_frames(pos, quat_mul(turn, quat))


def blend_human(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
  """Human frames `a` -> `b` [..., 168] by `alpha`: positions linear, orientations
  slerped."""
  pa, qa = split_frames(a)
  pb, qb = split_frames(b)
  return join_frames((1.0 - alpha) * pa + alpha * pb, slerp(qa, qb, alpha))


def human_stand() -> np.ndarray:
  """The neutral stand as a human frame [168]: the uniform skeleton in the SEED frame the
  robots' `stand.json` is retargeted from, its pelvis at x = y = 0 facing +x."""
  stand = json.loads(SKELETON_FILE.read_text())["stand"]
  return join_frames(np.asarray(stand["pos"]), np.asarray(stand["quat"]))
