"""Booster's official actuators for the T2 (the `t2` robot): the PD gains of
BoosterRobotics' `booster_deploy` (`booster_deploy/robots/t2.py`,
`T2_31DOF_CFG.joint_stiffness` / `joint_damping`), the gains their T2 dance and walk
policies run on the robot, on `booster_train`'s motor model (`robots.booster_motor`:
explicit PD with a two-sided torque-speed bound per motor, `actuator.py`'s
`BoosterJoint*` and parallel-joint wrappers).

Booster sets every joint by the BeyondMimic rule at 8 Hz, kp = armature * w^2 and
kd = 4 * armature * w with w = 2 * pi * 8 (`booster_train`, `BoosterJointCfg`,
`natural_freq` 8), on the same armatures as ours; the deploy config gives the parallel
waist pitch/roll and ankle pitch/roll the bare B6025 motor's damping (2.754549) instead
of the formula's on their reflected armature. Each joint's torque bound is its motor's
effort limit (`T2_JOINT_EFFORT_LIMIT`, Booster's) up to the motor's knee speed, then
falls linearly to zero at its no-load speed (`BOOSTER_MOTORS`; a parallel waist or ankle
joint scales both speeds by its wrapper's velocity ratio). The armatures, friction loss
and keyframe are unchanged, and so is the action-scale rule, 0.25 * effort / kp
(`booster_train`'s `T2_ACTION_SCALE`), now on these kp. Not modelled: `booster_train`'s
command delay (the env has its own 0-40 ms, the same 0-8 physics steps) and its x1.2 sim
effort limit (no effect on an explicit actuator, whose torque is already bounded at 1x).
The gains the T2 used before (10 Hz on the legs, kp 20 above, built-in
implicit PD) were not Booster's.
"""

from __future__ import annotations

import dataclasses
import math

if TYPE_CHECKING := False:  # pragma: no cover
  from mjlab.entity import EntityCfg

# Native joint pattern -> (kp, kd), `booster_deploy`'s T2_31DOF_CFG; both sides share a
# pattern.
BOOSTER_PD: dict[str, tuple[float, float]] = {
  "aa_head_yaw_joint": (18.214394, 1.449455),
  "head_pitch_joint": (18.214394, 1.449455),
  ".*_shoulder_pitch_joint": (42.952518, 3.418053),
  ".*_shoulder_roll_joint": (34.614676, 2.754549),
  ".*_elbow_pitch_joint": (34.614676, 2.754549),
  ".*_elbow_yaw_joint": (34.614676, 2.754549),
  ".*_wrist_pitch_joint": (18.214394, 1.449455),
  ".*_wrist_yaw_joint": (18.214394, 1.449455),
  ".*_wrist_roll_joint": (18.214394, 1.449455),
  "waist_pitch_joint": (99.837519, 2.754549),
  "waist_roll_joint": (132.424295, 2.754549),
  "waist_yaw_joint": (70.997986, 5.649840),
  ".*_hip_pitch_joint": (70.997986, 5.649840),
  ".*_hip_roll_joint": (70.997986, 5.649840),
  ".*_hip_yaw_joint": (70.997986, 5.649840),
  ".*_knee_pitch_joint": (70.997986, 5.649840),
  ".*_ankle_pitch_joint": (90.690730, 2.754549),
  ".*_ankle_roll_joint": (33.693992, 2.754549),
}


# Booster motor -> (no-load speed, knee speed), rpm (`booster_train` `BoosterJoint*`).
BOOSTER_MOTORS: dict[str, tuple[float, float]] = {
  "B4522-35": (177.0, 122.0),
  "B6030-24": (224.0, 110.0),
  "B6025-24": (227.0, 123.0),
  "B7629-22": (200.0, 59.0),
}
# Native joint pattern -> (motor, velocity ratio); the ratio is the parallel waist or
# ankle wrapper's (`BoosterT2WaistParaWrapperCfg` 0.5 / 0.4, `BoosterT2AnkleParaWrapperCfg`
# 0.7 / 1.0), on both speeds.
BOOSTER_JOINT_MOTOR: dict[str, tuple[str, float]] = {
  "aa_head_yaw_joint": ("B4522-35", 1.0),
  "head_pitch_joint": ("B4522-35", 1.0),
  ".*_shoulder_pitch_joint": ("B6030-24", 1.0),
  ".*_shoulder_roll_joint": ("B6025-24", 1.0),
  ".*_elbow_pitch_joint": ("B6025-24", 1.0),
  ".*_elbow_yaw_joint": ("B6025-24", 1.0),
  ".*_wrist_pitch_joint": ("B4522-35", 1.0),
  ".*_wrist_yaw_joint": ("B4522-35", 1.0),
  ".*_wrist_roll_joint": ("B4522-35", 1.0),
  "waist_pitch_joint": ("B6025-24", 0.5),
  "waist_roll_joint": ("B6025-24", 0.4),
  "waist_yaw_joint": ("B7629-22", 1.0),
  ".*_hip_pitch_joint": ("B7629-22", 1.0),
  ".*_hip_roll_joint": ("B7629-22", 1.0),
  ".*_hip_yaw_joint": ("B7629-22", 1.0),
  ".*_knee_pitch_joint": ("B7629-22", 1.0),
  ".*_ankle_pitch_joint": ("B6025-24", 0.7),
  ".*_ankle_roll_joint": ("B6025-24", 1.0),
}
_RPM = 2.0 * math.pi / 60.0


def get_t2_booster_robot_cfg() -> EntityCfg:
  """A fresh T2 entity cfg with one Booster motor actuator per joint pattern:
  Booster's deploy gains, the joint's effort limit, armature and friction loss, and
  its motor's torque-speed bound."""
  from morphmimic.robots.booster_motor import (
    BoosterMotorActuatorCfg,
    saturation_from_knee,
  )
  from morphmimic.robots.booster_t2.t2_constants import get_t2_robot_cfg

  base = get_t2_robot_cfg()
  assert base.articulation is not None
  actuators = []
  for a in base.articulation.actuators:
    (pattern,) = a.target_names_expr
    kp, kd = BOOSTER_PD[pattern]
    motor, ratio = BOOSTER_JOINT_MOTOR[pattern]
    v_max, knee = (ratio * rpm * _RPM for rpm in BOOSTER_MOTORS[motor])
    actuators.append(
      BoosterMotorActuatorCfg(
        target_names_expr=a.target_names_expr,
        stiffness=kp,
        damping=kd,
        effort_limit=a.effort_limit,
        saturation_effort=saturation_from_knee(a.effort_limit, v_max, knee),
        velocity_limit=v_max,
        armature=a.armature,
        frictionloss=a.frictionloss,
      )
    )
  articulation = dataclasses.replace(base.articulation, actuators=tuple(actuators))
  return dataclasses.replace(base, articulation=articulation)


def booster_action_scale() -> dict[str, float]:
  """0.25 * effort / kp per joint pattern, on Booster's kp."""
  from morphmimic.robots.booster_t2.t2_constants import T2_JOINT_EFFORT_LIMIT

  return {
    pattern: 0.25 * T2_JOINT_EFFORT_LIMIT[pattern] / kp
    for pattern, (kp, _) in BOOSTER_PD.items()
  }
