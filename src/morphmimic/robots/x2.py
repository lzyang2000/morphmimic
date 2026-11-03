"""AgiBot X2 Ultra spec: 29 body joints + 2 head joints, wrist yaw/roll swapped; the
actuators are AgiBot's official RL plant (`robots.x2_agibot`)."""

from __future__ import annotations

import math

from morphmimic.robots.spec import RobotSpec, Translator, height_scale

# The 31 hinges of x2_ultra.xml in model order.
X2_JOINT_NAMES: tuple[str, ...] = tuple(
  f"{j}_joint"
  for j in (
    *(
      f"{side}_{j}"
      for side in ("left", "right")
      for j in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
      )
    ),
    "waist_yaw",
    "waist_pitch",
    "waist_roll",
    *(
      f"{side}_{j}"
      for side in ("left", "right")
      for j in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
      )
    ),
    "head_yaw",
    "head_pitch",
  )
)

# Canonical (G1) name -> X2 name; unlisted joints keep their name.
_CANONICAL_TO_NATIVE = {
  f"{side}_{a}": f"{side}_{b}"
  for side in ("left", "right")
  for a, b in (
    ("wrist_roll_joint", "wrist_yaw_joint"),
    ("wrist_yaw_joint", "wrist_roll_joint"),
  )
}
# Signs and offsets, checked by forward kinematics (native = sign * q + offset):
# - elbow: offset -pi/2, the G1's zero has the forearm forward (elbow bent 90 deg), the
#   X2's has it straight; sign +1, both flex as the native angle goes negative.
# - wrist roll (X2 wrist yaw): -1, with the forearm forward the X2's twist axis points
#   back toward the elbow.
_SIGN = {f"{side}_wrist_roll_joint": -1.0 for side in ("left", "right")}
_OFFSET = {f"{side}_elbow_joint": -math.pi / 2 for side in ("left", "right")}


def x2_spec() -> RobotSpec:
  from morphmimic.constants import TRACKED_BODY_NAMES
  from morphmimic.robots.agibot_x2_ultra.x2_ultra_constants import (
    X2_ULTRA_BEYONDMIMIC_KPKD_10HZ_ACTION_SCALE,
    X2_ULTRA_FOOT_COLLISION_EXPR,
    get_x2_robot_cfg,
  )

  from morphmimic.robots.x2_agibot import agibot_action_scale, get_x2_agibot_robot_cfg

  hands = {f"{s}_wrist_yaw_link": f"{s}_wrist_roll_link" for s in ("left", "right")}
  head = {
    k: v for k, v in X2_ULTRA_BEYONDMIMIC_KPKD_10HZ_ACTION_SCALE.items()
    if k.startswith("head")
  }
  return RobotSpec(
    name="x2",
    entity_cfg_fn=get_x2_agibot_robot_cfg,
    action_scale=agibot_action_scale(head),
    joint_names=X2_JOINT_NAMES,
    root_body="pelvis",
    anchor_body="torso_link",
    imu_body="pelvis",
    tracked_bodies=tuple(hands.get(b, b) for b in TRACKED_BODY_NAMES),
    foot_geom_regex=X2_ULTRA_FOOT_COLLISION_EXPR,
    com_body="torso_link",
    translator=Translator.from_names(
      X2_JOINT_NAMES, _CANONICAL_TO_NATIVE, _SIGN, _OFFSET
    ),
    csv_joint_columns=X2_JOINT_NAMES,
    height_scale=height_scale(get_x2_robot_cfg, "left_knee_link"),
    contact_budget=(96, 1024),
    zmin_exclude_regex=r".*_foot_contact_point$",
  )
