"""Observation terms of the SuperMimic env.

Every term is canonical (29 joints, 14 tracked bodies) for every robot: joint terms go
through the robot's translator, read from the motion command (`cmd.spec.translator`),
and the proprioceptive gyro and gravity come from tracked slot 0's body state (spec
section 8.1), not from an IMU site or the floating base.

`wm_state`, `goal_future` and `goal_human` give the robot state and the goals in
the motion library's frame (world positions minus the env origin), so they can be
compared directly with `MotionLibrary.state` / `future` and fed to the analytic heads of
`morphmimic.model.rewards_torch`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from torch import Tensor

from morphmimic.constants import HUMAN_ENV_DIM
from morphmimic.env.commands import MultiMotionCommand
from morphmimic.model.human import pack_human_goal
from morphmimic.model.rewards_torch import goal_features
from morphmimic.model.state import StateParts, anchor_quat, join_state

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


def _command(env: ManagerBasedRlEnv, command_name: str) -> MultiMotionCommand:
  return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


def _robot_state(env: ManagerBasedRlEnv, cmd: MultiMotionCommand) -> Tensor:
  return join_state(
    StateParts(
      joint_pos=cmd.robot_joint_pos,
      joint_vel=cmd.robot_joint_vel,
      body_pos=cmd.robot_body_pos_w - env.scene.env_origins[:, None, :],
      body_quat=cmd.robot_body_quat_w,
      body_lin_vel=cmd.robot_body_lin_vel_w,
      body_ang_vel=cmd.robot_body_ang_vel_w,
    )
  )


def wm_state(env: ManagerBasedRlEnv, command_name: str) -> Tensor:
  """World-model state s [B, 240] of the robot's tracked bodies."""
  return _robot_state(env, _command(env, command_name))


def goal_future(env: ManagerBasedRlEnv, command_name: str) -> Tensor:
  """Goal [B, K * 67] from the command's next K reference frames, in the command's
  `goal_anchor` frame."""
  cmd = _command(env, command_name)
  return goal_features(_robot_state(env, cmd), cmd.future_states(), cmd.cfg.goal_anchor)


def goal_human(env: ManagerBasedRlEnv, command_name: str) -> Tensor:
  """Raw human goal [B, 1684] (`pack_human_goal`): the command's next K human poses
  and the robot anchor body's (torso_link) quaternion (wxyz, world, `anchor_quat`).
  Zeros when the library has no human sidecars."""
  cmd = _command(env, command_name)
  if not cmd.library.has_human:
    return torch.zeros(env.num_envs, HUMAN_ENV_DIM, device=env.device)
  window = cmd.human_future()
  robot_quat = anchor_quat(_robot_state(env, cmd))
  return pack_human_goal(window[..., :3], window[..., 3:], robot_quat)


def imu_ang_vel(env: ManagerBasedRlEnv) -> Tensor:
  """IMU gyro sensor [B, 3]. Not used by the env cfg (`canonical_ang_vel` replaces
  it); kept for the pre-registry G1 cfg of the value regression test."""
  return builtin_sensor(env, "robot/imu_ang_vel")


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def canonical_joint_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> Tensor:
  """Encoder-biased joint positions relative to the default pose, canonical [B, 29]."""
  tr = _command(env, command_name).spec.translator
  asset: Entity = env.scene[asset_cfg.name]
  return tr.to_canonical(asset.data.joint_pos_biased) - tr.to_canonical(
    asset.data.default_joint_pos
  )


def canonical_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> Tensor:
  """Joint velocities, canonical [B, 29]."""
  tr = _command(env, command_name).spec.translator
  asset: Entity = env.scene[asset_cfg.name]
  return tr.to_canonical_delta(asset.data.joint_vel)


def _imu(
  env: ManagerBasedRlEnv, command_name: str, imu_slot: int | None = None
) -> tuple[Tensor, Tensor]:
  """World quaternion and angular velocity of the robot's IMU body (`RobotSpec.
  imu_body`: the G1 and X2 pelvis, the T2 trunk), or of tracked slot `imu_slot`."""
  cmd = _command(env, command_name)
  body = cmd.body_indexes[cmd.spec.imu_slot if imu_slot is None else imu_slot]
  data = cmd.robot.data
  return data.body_link_quat_w[:, body], data.body_link_ang_vel_w[:, body]


def canonical_ang_vel(
  env: ManagerBasedRlEnv, command_name: str, imu_slot: int | None = None
) -> Tensor:
  """Gyro [B, 3]: the IMU body's angular velocity in its own frame."""
  quat, ang_vel = _imu(env, command_name, imu_slot)
  return quat_apply_inverse(quat, ang_vel)


def canonical_gravity(
  env: ManagerBasedRlEnv, command_name: str, imu_slot: int | None = None
) -> Tensor:
  """Gravity direction [B, 3] in the IMU body's frame."""
  quat, _ = _imu(env, command_name, imu_slot)
  return quat_apply_inverse(quat, _command(env, command_name).robot.data.gravity_vec_w)


class sim_context:
  """Context xi [B, 33] = [foot friction | encoder bias (29) | CoM offset (3)].

  Friction is axis 0 of the first geom matching `foot_geom_regex` (the DR event
  shares one value across the foot geoms); the encoder bias is canonical
  (`to_canonical_delta` of the native bias, translator of the command
  `command_name`); the CoM offset is the current `body_ipos` of `com_body_name`
  minus its compiled default. Model ids are resolved once, at construction
  (class-based term).
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    params = cfg.params
    asset: Entity = env.scene[params["asset_cfg"].name]
    # The command manager is built before the observation manager.
    self._translator = _command(env, params["command_name"]).spec.translator
    geom = asset.find_geoms(params["foot_geom_regex"])[0][0]
    body = asset.find_bodies(params["com_body_name"])[0][0]
    self._geom_id = int(asset.indexing.geom_ids[geom])
    self._body_id = int(asset.indexing.body_ids[body])
    self._default_ipos = torch.as_tensor(
      env.sim.mj_model.body_ipos[self._body_id],
      dtype=torch.float32,
      device=env.device,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    foot_geom_regex: str,
    com_body_name: str,
  ) -> Tensor:
    del command_name, foot_geom_regex, com_body_name  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    friction = env.sim.model.geom_friction[:, self._geom_id, 0].expand(env.num_envs)
    com_offset = env.sim.model.body_ipos[:, self._body_id] - self._default_ipos
    return torch.cat(
      [
        friction.unsqueeze(-1),
        self._translator.to_canonical_delta(asset.data.encoder_bias),
        com_offset.expand(env.num_envs, 3),
      ],
      dim=-1,
    )
