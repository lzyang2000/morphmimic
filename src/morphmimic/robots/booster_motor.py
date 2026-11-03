"""Booster's motor model (`booster_train`, `BoosterDelayedPDActuator._clip_effort`):
explicit PD whose torque bound falls with joint speed alike in both directions, the
effort limit up to the knee speed, then linearly to zero at the no-load speed.

mjlab's `DcMotorActuator` has the same line in the motoring quadrant but lets a joint
brake at full torque at any speed; Booster bounds both directions by |v|. The line is
written in mjlab's terms, stall torque S and no-load speed v_max, with
S = effort * v_max / (v_max - v_knee) (`saturation_from_knee`), so the actuator keeps
mjlab's per-target parameters and its stateless compute: it stays fusable, and the
env's command delay applies as to any mjlab actuator (`booster_train`'s own 0-8
physics-step delay is not modelled here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg
from mjlab.actuator.pd_actuator import pd_torque

if TYPE_CHECKING:
  from mjlab.entity import Entity


def booster_motor_clip(
  effort: torch.Tensor,
  saturation_effort: torch.Tensor,
  velocity_limit: torch.Tensor,
  force_limit: torch.Tensor,
  vel: torch.Tensor,
) -> torch.Tensor:
  """`effort` within +-min(force_limit, max(0, S (1 - |vel| / v_max)))."""
  bound = (saturation_effort * (1.0 - vel.abs() / velocity_limit)).clamp(min=0.0)
  bound = torch.minimum(bound, force_limit)
  return torch.clamp(effort, -bound, bound)


def saturation_from_knee(effort: float, velocity_limit: float, knee: float) -> float:
  """Stall torque of the line through (knee, effort) and (velocity_limit, 0)."""
  if not 0.0 <= knee < velocity_limit:
    raise ValueError(f"need 0 <= knee < velocity_limit, got {knee}, {velocity_limit}")
  return effort * velocity_limit / (velocity_limit - knee)


@dataclass(kw_only=True)
class BoosterMotorActuatorCfg(DcMotorActuatorCfg):
  """`DcMotorActuatorCfg` with Booster's two-sided torque-speed bound."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BoosterMotorActuator:
    return BoosterMotorActuator(self, entity, target_ids, target_names)


class BoosterMotorActuator(DcMotorActuator[BoosterMotorActuatorCfg]):
  """`DcMotorActuator` with `booster_motor_clip` as its control law's bound."""

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    torque = pd_torque(params["stiffness"], params["damping"], cmd)
    return booster_motor_clip(
      torque,
      params["saturation_effort"],
      params["velocity_limit_motor"],
      params["force_limit"],
      cmd.vel,
    )
