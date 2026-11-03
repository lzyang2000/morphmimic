"""AgiBot's official RL plant for the X2 Ultra (the `x2` robot), from the AimDK_X2 extra
`aimdk-extra-x2_rl_deploy-v1.1` (`x2_rl_deploy_controller/config/motion_control.yaml`
and the MJCF `x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml`).

Every body joint is a built-in (implicit) position PD with AgiBot's policy gains
(`rl_config.kps` / `kds`), its MJCF's motor torque limit (`ctrlrange`), armature 0.03
and friction loss 0.3 (the MJCF's joint default); the action scale is AgiBot's flat
0.25. The head keeps the x2_teleop actuators with the same armature and friction. The
MJCF's kinematics and inertias equal `x2_ultra.xml`'s (checked 2026-10-26), so the
model, keyframe, collisions and the reference motions are unchanged; only the
actuators differ from the x2_teleop DC-motor plant (d11f332) the X2 used before.
"""

from __future__ import annotations

import dataclasses

if TYPE_CHECKING := False:  # pragma: no cover
  from mjlab.entity import EntityCfg

# Native joint pattern -> (kp, kd, torque limit N m); both sides share a pattern.
AGIBOT_PD: dict[str, tuple[float, float, float]] = {
  ".*_hip_pitch_joint": (120.0, 5.0, 118.0),
  ".*_hip_roll_joint": (100.0, 4.0, 118.0),
  ".*_hip_yaw_joint": (100.0, 4.0, 118.0),
  ".*_knee_joint": (150.0, 5.0, 118.0),
  ".*_ankle_pitch_joint": (40.0, 2.0, 36.0),
  ".*_ankle_roll_joint": (40.0, 2.0, 24.0),
  "waist_yaw_joint": (40.1792, 2.5579, 118.0),
  "waist_pitch_joint": (200.0, 2.0, 48.0),
  "waist_roll_joint": (200.0, 2.0, 48.0),
  ".*_shoulder_pitch_joint": (50.0, 3.0, 36.0),
  ".*_shoulder_roll_joint": (50.0, 3.0, 36.0),
  ".*_shoulder_yaw_joint": (50.0, 3.0, 24.0),
  ".*_elbow_joint": (50.0, 3.0, 24.0),
  ".*_wrist_yaw_joint": (20.0, 2.0, 24.0),
  ".*_wrist_pitch_joint": (20.0, 2.0, 2.2),
  ".*_wrist_roll_joint": (20.0, 2.0, 2.2),
}
AGIBOT_ACTION_SCALE = 0.25
AGIBOT_ARMATURE = 0.03
AGIBOT_FRICTIONLOSS = 0.3


def get_x2_agibot_robot_cfg() -> EntityCfg:
  """A fresh X2 entity cfg with one built-in PD actuator per AgiBot joint group."""
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg

  from morphmimic.robots.agibot_x2_ultra.x2_ultra_constants import get_x2_robot_cfg

  base = get_x2_robot_cfg()
  assert base.articulation is not None
  body = tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(pattern,),
      stiffness=kp,
      damping=kd,
      effort_limit=effort,
      armature=AGIBOT_ARMATURE,
      frictionloss=AGIBOT_FRICTIONLOSS,
    )
    for pattern, (kp, kd, effort) in AGIBOT_PD.items()
  )
  head = tuple(
    dataclasses.replace(a, armature=AGIBOT_ARMATURE, frictionloss=AGIBOT_FRICTIONLOSS)
    for a in base.articulation.actuators
    if all(e.startswith("head") for e in a.target_names_expr)
  )
  if not head:
    raise ValueError("expected the X2's head actuators")
  articulation = dataclasses.replace(base.articulation, actuators=body + head)
  return dataclasses.replace(base, articulation=articulation)


def agibot_action_scale(head: dict[str, float]) -> dict[str, float]:
  """AgiBot's flat action scale on the body joints, `head` (pattern -> scale) on the head."""
  return {**dict.fromkeys(AGIBOT_PD, AGIBOT_ACTION_SCALE), **head}
