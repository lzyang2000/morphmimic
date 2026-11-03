"""MorphMimic flat tracking env cfg for any registered robot:
mjlab's BeyondMimic flat tracking cfg (`make_tracking_env_cfg`) with the
robot's entity from `morphmimic.robots.get_robot`, a library-backed motion command,
the policy, critic and goal observation groups, no push event, no self-collision sensor/penalty,
joint-limit weight -2 and no end-effector termination. A `clip_end` time-out ends the
episode at the last frame of its clip, so the robot is never moved mid-episode.

Everything the trainer sees is canonical (29 joints, 14 tracked bodies) for every
robot: a canonical action term (`actions.CanonicalJointPositionActionCfg`), canonical
joint observation terms, and the gyro and gravity of tracked slot 0's body state.
For the G1 (identity translator) the values equal the pre-registry cfg's.

Floor friction: MuJoCo uses the larger friction of two contacting geoms (equal
priority), so with mjlab's floor (1.0) the foot-friction DR in [0.3, 1.2] would barely
change the contact. The floor is set to 0.3, making the effective friction the
randomized foot value that the context reports. In eval (no DR, foot default 0.6) the
floor is 1.0, giving the paper's evaluation friction of 1.0.

Run it with `MorphMimicEnv` (rewards and terminations on post-step kinematics) and
`auto_reset=False`: the caller resets finished envs with `env.reset(env_ids=...)`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import fields, replace

from mjlab.actuator import ActuatorCfg, BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import GeomCfg
from mjlab.utils.string import resolve_expr
from torch import Tensor

from morphmimic.constants import OBS_NOISE, GoalAnchor
from morphmimic.env import observations as obs
from morphmimic.env import terminations
from morphmimic.env.actions import CanonicalJointPositionActionCfg
from morphmimic.env.commands import MultiMotionCommandCfg
from morphmimic.robots import RobotSpec, get_robot

COMMAND_NAME = "motion"
# Name of the plane terrain's geom.
FLOOR_GEOM_NAME = "terrain"
FLOOR_FRICTION = 0.3
EVAL_FLOOR_FRICTION = 1.0
ACTUATOR_DELAY_MAX_S = 0.02
# Lag resampling period (physics steps) of a fixed delay: longer than any episode, so
# the lag drawn on an episode's first step (mjlab's `DelayBuffer` resets the step
# counter) holds until the next reset.
FIXED_DELAY_PERIOD = 2**40


def _proprio_terms(
  noisy: bool, imu_slot: int | None = None
) -> dict[str, ObservationTermCfg]:
  """Policy observation terms, in `analytic_obs` order (Table A.1 noise if noisy):
  canonical joints, then the IMU body's gyro and gravity (tracked slot `imu_slot`'s
  if given), then the canonical last action."""

  def term(
    func: Callable[..., Tensor], key: str, **extra: int | None
  ) -> ObservationTermCfg:
    noise = Unoise(n_min=-OBS_NOISE[key], n_max=OBS_NOISE[key]) if noisy else None
    params = {"command_name": COMMAND_NAME, **extra}
    return ObservationTermCfg(func=func, params=params, noise=noise)

  return {
    "joint_pos": term(obs.canonical_joint_pos, "joint_pos"),
    "joint_vel": term(obs.canonical_joint_vel, "joint_vel"),
    "ang_vel": term(obs.canonical_ang_vel, "ang_vel", imu_slot=imu_slot),
    "gravity": term(obs.canonical_gravity, "gravity", imu_slot=imu_slot),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }


def _group(
  terms: dict[str, ObservationTermCfg], corruption: bool = False
) -> ObservationGroupCfg:
  return ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=corruption
  )


def _observations(
  spec: RobotSpec, imu_slot: int | None = None
) -> dict[str, ObservationGroupCfg]:
  command = {"command_name": COMMAND_NAME}
  context = {
    "asset_cfg": SceneEntityCfg("robot"),
    "command_name": COMMAND_NAME,
    "foot_geom_regex": spec.foot_geom_regex,
    "com_body_name": spec.com_body,
  }
  return {
    "policy": _group(_proprio_terms(noisy=True, imu_slot=imu_slot), corruption=True),
    "policy_clean": _group(_proprio_terms(noisy=False, imu_slot=imu_slot)),
    "goal": _group({"goal": ObservationTermCfg(func=obs.goal_future, params=command)}),
    "goal_kp": _group(
      {"goal_kp": ObservationTermCfg(func=obs.goal_human, params=command)}
    ),
    "state": _group({"state": ObservationTermCfg(func=obs.wm_state, params=command)}),
    "context": _group(
      {"context": ObservationTermCfg(func=obs.sim_context, params=context)}
    ),
  }


def _as_builtin(actuator: ActuatorCfg) -> ActuatorCfg:
  """A DC motor as built-in position PD with the same gains, limits, armature and
  friction (its saturation and velocity limits dropped); any other actuator as is."""
  if not isinstance(actuator, DcMotorActuatorCfg):
    return actuator
  return BuiltinPositionActuatorCfg(
    **{f.name: getattr(actuator, f.name) for f in fields(BuiltinPositionActuatorCfg)}
  )


def policy_joint_frame(
  target: RobotSpec, source: RobotSpec
) -> tuple[dict[str, float], dict[str, float]]:
  """Per native joint of `target`, the action scale and default joint position that
  put `source`'s joint frame on it: in every canonical slot both equal
  `source`'s (the default pose taken through both translators), so a canonical action
  or joint observation means the same angle on both bodies. Untracked joints (the
  heads) keep `target`'s own. Keys are the exact native joint names."""

  def per_joint(spec: RobotSpec) -> tuple[list[float], list[float]]:
    _, names, scale = resolve_matching_names_values(
      spec.action_scale, list(spec.joint_names)
    )
    if tuple(names) != spec.joint_names:
      raise ValueError(f"{spec.name}: action scale does not cover every joint")
    init = spec.entity_cfg_fn().init_state.joint_pos
    return list(scale), list(resolve_expr(init, spec.joint_names, 0.0))

  src_scale, src_default = per_joint(source)
  scale, default = per_joint(target)
  src_tr, tr = source.translator, target.translator
  for slot, (s, n) in enumerate(zip(src_tr.perm, tr.perm, strict=True)):
    q_canon = src_tr.sign[slot] * (src_default[s] - src_tr.offset[slot])
    scale[n] = src_scale[s]
    default[n] = tr.sign[slot] * q_canon + tr.offset[slot]
  return dict(zip(target.joint_names, scale)), dict(zip(target.joint_names, default))


def morphmimic_env_cfg(
  robot: str,
  subset_yaml: str = "",
  num_envs: int = 4096,
  play: bool = False,
  eval: bool = False,
  actuator_delay: bool = True,
  actuator_delay_max_s: float = ACTUATOR_DELAY_MAX_S,
  actuator_delay_fixed: bool = True,
  builtin_pd: bool = False,
  joint_frame: str | None = None,
  imu_body: str | None = None,
  human_root: str | None = None,
  goal_anchor: GoalAnchor = "reference",
) -> ManagerBasedRlEnvCfg:
  """Env cfg of `robot` (a `morphmimic.robots` registry name) over the clips of
  `subset_yaml` (the library is built with the env; its clips must be that robot's).

  play: mjlab's play overrides (endless episodes, no observation noise, no RSI pose /
  velocity noise, every env starts at frame 0 of its clip). eval: no DR,
  no observation or RSI noise, floor friction 1.0, no episode time limit (the
  `clip_end` time-out ends each clip), start at frame 0 of the clip set per env with
  `MultiMotionCommand.assign_clips`.

  actuator_delay: random position-target delay of 0 to `actuator_delay_max_s`
  (default 20 ms, a multiple of the 5 ms physics step), drawn per env by mjlab's
  actuator buffers. Disabled in nominal eval mode.

  actuator_delay_fixed: draw the delay once per episode (on its first physics step)
  and hold it (default: a real robot's latency is steady); False: redraw it every
  physics step.

  builtin_pd: drive every DC-motor joint (the T2's Booster motors) with MuJoCo's built-in implicit PD
  instead, same gains, effort limit, armature and friction. No change on the G1 and
  on the X2's AgiBot plant (built-in already).

  joint_frame: registry name of the robot whose action scale and default pose the
  canonical joints use (`policy_joint_frame`), for running a policy of that robot on
  this body; None: this robot's own.

  imu_body: a tracked body the policy's gyro and gravity are read on instead of the
  robot's IMU body (a sim-only check, e.g. the T2's pelvis-equivalent slot 0 for a
  policy trained on a pelvis IMU); None: `spec.imu_body`.

  human_root: root of the clips' human sidecars (`MultiMotionCommandCfg.human_root`);
  None: `<clip root>/human50`. Without sidecars the `goal_kp` group is zeros.

  goal_anchor: frame of the motion goal's anchor position
  (`MultiMotionCommandCfg.goal_anchor`, `rewards_torch.goal_features`).
  """
  spec = get_robot(robot)
  imu_slot = None
  if imu_body is not None:
    if imu_body not in spec.tracked_bodies:
      raise ValueError(f"{robot}: {imu_body!r} is not a tracked body")
    imu_slot = spec.tracked_bodies.index(imu_body)
  cfg = make_tracking_env_cfg()
  entity = spec.entity_cfg_fn()
  action_scale = spec.action_scale
  if joint_frame is not None:
    action_scale, default = policy_joint_frame(spec, get_robot(joint_frame))
    entity.init_state = replace(entity.init_state, joint_pos=default)
  if builtin_pd:
    assert entity.articulation is not None
    entity.articulation = replace(
      entity.articulation,
      actuators=tuple(_as_builtin(a) for a in entity.articulation.actuators),
    )
  cfg.scene.entities = {"robot": entity}
  cfg.scene.num_envs = num_envs
  cfg.scene.sensors = ()  # No self-collision sensor.

  cfg.actions = {
    "joint_pos": CanonicalJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=action_scale,
      use_default_offset=True,
      robot_name=robot,
    )
  }

  max_lag = 0
  if actuator_delay and not eval:
    max_lag = round(actuator_delay_max_s / cfg.sim.mujoco.timestep)
    if not math.isclose(max_lag * cfg.sim.mujoco.timestep, actuator_delay_max_s):
      raise ValueError(
        f"actuator delay {actuator_delay_max_s} s must be a multiple of the physics step"
      )
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  articulation.actuators = tuple(
    replace(
      actuator,
      delay_min_lag=0,
      delay_max_lag=max_lag,
      delay_hold_prob=0.0,
      delay_update_period=FIXED_DELAY_PERIOD if actuator_delay_fixed else 0,
      # A random phase would put the first draw anywhere in the period.
      delay_per_env_phase=not actuator_delay_fixed,
    )
    for actuator in articulation.actuators
  )

  mjlab_cmd = cfg.commands[COMMAND_NAME]
  assert isinstance(mjlab_cmd, MotionCommandCfg)
  cmd = MultiMotionCommandCfg(
    robot_name=robot,
    subset_yaml=subset_yaml,
    entity_name=mjlab_cmd.entity_name,
    anchor_body_name=spec.anchor_body,
    body_names=spec.tracked_bodies,
    resampling_time_range=mjlab_cmd.resampling_time_range,
    debug_vis=mjlab_cmd.debug_vis,
    pose_range=dict(mjlab_cmd.pose_range),
    velocity_range=dict(mjlab_cmd.velocity_range),
    joint_position_range=mjlab_cmd.joint_position_range,
    human_root=human_root,
    goal_anchor=goal_anchor,
  )
  cfg.commands = {COMMAND_NAME: cmd}
  cfg.observations = _observations(spec, imu_slot)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = spec.foot_geom_regex
  cfg.events["base_com"].params["asset_cfg"].body_names = (spec.com_body,)
  cfg.events.pop("push_robot")
  cfg.rewards.pop("self_collisions")
  cfg.rewards["joint_limit"].weight = -2.0
  cfg.terminations.pop("ee_body_pos")
  cfg.terminations["clip_end"] = TerminationTermCfg(
    func=terminations.clip_end,
    params={"command_name": COMMAND_NAME},
    time_out=True,
  )
  cfg.episode_length_s = 10.0
  cfg.auto_reset = False
  cfg.viewer.body_name = spec.anchor_body
  cfg.sim.nconmax, cfg.sim.njmax = spec.contact_budget

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cmd.pose_range = {}
    cmd.velocity_range = {}
    cmd.sampling = "start"

  floor_friction = FLOOR_FRICTION
  if eval:
    floor_friction = EVAL_FLOOR_FRICTION
    cfg.events = {}
    cfg.observations["policy"].enable_corruption = False
    cfg.episode_length_s = 1e9
    cmd.pose_range = {}
    cmd.velocity_range = {}
    cmd.joint_position_range = (0.0, 0.0)
    cmd.sampling = "start"
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.geoms = (
    GeomCfg(geom_names_expr=(FLOOR_GEOM_NAME,), friction=(floor_friction,)),
  )

  return cfg


def morphmimic_g1_env_cfg(
  subset_yaml: str = "",
  num_envs: int = 4096,
  play: bool = False,
  eval: bool = False,
  actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """`morphmimic_env_cfg("g1", ...)`."""
  return morphmimic_env_cfg(
    "g1",
    subset_yaml=subset_yaml,
    num_envs=num_envs,
    play=play,
    eval=eval,
    actuator_delay=actuator_delay,
  )
