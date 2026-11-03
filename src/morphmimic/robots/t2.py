"""Booster T2 spec: trunk-rooted, 29 body joints + 2 head joints, T-pose arm zero; the
actuators carry Booster's official deploy gains (`robots.t2_booster`).

Map correction (checked by forward kinematics): with the arm hanging and the forearm
forward, as at the G1's zero, the T2 wrist yaw bends about the G1 wrist pitch axis and
the T2 wrist roll about the G1 wrist yaw axis. So canonical wrist pitch -> T2 wrist yaw
and canonical wrist yaw -> T2 wrist roll, and the wrist chain is a plain permutation
(twist, pitch, yaw as on the G1), not a reversed bend order.
"""

from __future__ import annotations

import math

from morphmimic.robots.spec import RobotSpec, Translator, height_scale

T2_TRACKED_BODY_NAMES: tuple[str, ...] = (
  "waist_yaw_link",
  "left_hip_roll_link",
  "left_knee_pitch_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_pitch_link",
  "right_ankle_roll_link",
  "trunk",
  "left_shoulder_roll_link",
  "left_elbow_yaw_link",
  "left_wrist_roll_link",
  "right_shoulder_roll_link",
  "right_elbow_yaw_link",
  "right_wrist_roll_link",
)

# Canonical (G1) name -> T2 name by function; unlisted joints keep their name.
_CANONICAL_TO_NATIVE = {
  f"{side}_{a}": f"{side}_{b}"
  for side in ("left", "right")
  for a, b in (
    ("knee_joint", "knee_pitch_joint"),
    ("shoulder_yaw_joint", "elbow_pitch_joint"),
    ("elbow_joint", "elbow_yaw_joint"),
    ("wrist_roll_joint", "wrist_pitch_joint"),
    ("wrist_pitch_joint", "wrist_yaw_joint"),
    ("wrist_yaw_joint", "wrist_roll_joint"),
  )
}
# Signs and offsets, checked by forward kinematics (native = sign * q + offset).
# Read at the canonical zero: arm hanging (shoulder roll offset), forearm forward.
# - waist: -1, the XML parent of the waist chain is the trunk, not the pelvis.
# - shoulder roll: -/+pi/2, the T2's zero is the T-pose, the G1's hangs the arms.
# - shoulder yaw (elbow pitch): left -1, its axis points down the hanging left humerus.
# - elbow (elbow yaw): left offset -pi/2, +1; right offset +pi/2, -1. The G1's zero is
#   the elbow bent 90 deg, the T2's straight; left flexes negative, right positive.
# - right wrist roll and pitch: -1, mirrored axes once the right forearm points forward.
_SIGN = {
  "waist_yaw_joint": -1.0,
  "waist_roll_joint": -1.0,
  "waist_pitch_joint": -1.0,
  "left_shoulder_yaw_joint": -1.0,
  "right_elbow_joint": -1.0,
  "right_wrist_roll_joint": -1.0,
  "right_wrist_pitch_joint": -1.0,
}
_OFFSET = {
  "left_shoulder_roll_joint": -math.pi / 2,
  "right_shoulder_roll_joint": math.pi / 2,
  "left_elbow_joint": -math.pi / 2,
  "right_elbow_joint": math.pi / 2,
}


def t2_spec() -> RobotSpec:
  from morphmimic.robots.booster_t2.t2_constants import (
    T2_ACTION_JOINT_NAMES,
    T2_FOOT_COLLISION_EXPR,
    get_t2_robot_cfg,
  )
  from morphmimic.robots.t2_booster import (
    booster_action_scale,
    get_t2_booster_robot_cfg,
  )

  return RobotSpec(
    name="t2",
    entity_cfg_fn=get_t2_booster_robot_cfg,
    action_scale=booster_action_scale(),
    joint_names=T2_ACTION_JOINT_NAMES,
    root_body="trunk",
    anchor_body="trunk",
    imu_body="trunk",
    tracked_bodies=T2_TRACKED_BODY_NAMES,
    foot_geom_regex=T2_FOOT_COLLISION_EXPR,
    com_body="trunk",
    translator=Translator.from_names(
      T2_ACTION_JOINT_NAMES, _CANONICAL_TO_NATIVE, _SIGN, _OFFSET
    ),
    csv_joint_columns=T2_ACTION_JOINT_NAMES,
    height_scale=height_scale(get_t2_robot_cfg, "left_knee_pitch_link"),
    contact_budget=(64, 400),
  )
