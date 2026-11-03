"""Canonical joint-position action (spec section 4.2).

The policy acts in the canonical 29-joint space for every robot. The term maps a
canonical action onto the robot's native actuators through its translator: native
target = default pose + `to_native_delta(a * scale)`, with each canonical slot scaled
by the scale of the native actuator it drives. Untracked joints (the X2/T2 heads) get
no delta, so they are held at their default pose. For the G1 (identity translator)
this is mjlab's `JointPositionAction`, value for value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from morphmimic.constants import NUM_JOINTS
from morphmimic.robots import get_robot

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class CanonicalJointPositionAction(JointPositionAction):
  """`JointPositionAction` with a canonical [N, 29] raw action.

  `raw_action` (and so mjlab's `last_action` and the action manager's `action`) is
  the canonical action; `_processed_actions` stays at native width [N, J] and
  `apply_actions` (inherited) subtracts the encoder bias from it.
  """

  cfg: CanonicalJointPositionActionCfg

  def __init__(self, cfg: CanonicalJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    if not cfg.use_default_offset:
      raise ValueError("CanonicalJointPositionAction needs use_default_offset=True.")
    tr = get_robot(cfg.robot_name).translator
    # Action index == native joint index: the translator indexes native joints.
    if self._target_ids.tolist() != list(range(tr.num_native)):
      raise ValueError(
        f"{cfg.robot_name}: the action must drive every native joint in entity order;"
        f" got targets {self._target_names}."
      )
    self._perm = torch.as_tensor(tr.perm, dtype=torch.long, device=self.device)
    self._sign = torch.as_tensor(tr.sign, dtype=torch.float32, device=self.device)
    scale = torch.as_tensor(self._scale, dtype=torch.float32, device=self.device)
    self._scale_canon = scale.expand(self.num_envs, tr.num_native)[:, self._perm]
    self._action_dim = NUM_JOINTS
    self._raw_actions = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
    # Exploration noise executed here (`cfg.noise_std`, scaled per env by
    # `noise_scale`), so that mjlab's `last_action` stays the clean action while the
    # robot runs `executed` = a + eps (the collector logs it for the world model).
    self.noise_std = cfg.noise_std
    self.noise_scale = torch.ones(self.num_envs, 1, device=self.device)
    self.executed = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    """Canonical action [N, 29] -> native targets: default + to_native_delta(a' * s),
    with a' = a + noise_std * noise_scale * N(0, 1) (`executed`); `raw_action` is a."""
    self._raw_actions[:] = actions
    if self.noise_std > 0.0:
      self.executed[:] = actions + self.noise_std * self.noise_scale * torch.randn_like(
        actions
      )
    else:
      self.executed[:] = actions
    delta = torch.zeros_like(self._processed_actions)
    delta[:, self._perm] = self._sign * self.executed * self._scale_canon
    self._processed_actions = delta + self._offset
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
      )


@dataclass(kw_only=True)
class CanonicalJointPositionActionCfg(JointPositionActionCfg):
  robot_name: str
  """Registry name of the robot (`morphmimic.robots.get_robot`)."""
  noise_std: float = 0.0
  """Std of the N(0, 1) noise the term adds to the action it executes (the recorded
  action stays clean); the collector sets it for `--clean-action-record`. 0: none."""

  def build(self, env: ManagerBasedRlEnv) -> CanonicalJointPositionAction:
    return CanonicalJointPositionAction(self, env)
