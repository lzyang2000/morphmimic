"""G1 Amber (`g1_amber`): the G1 with the waist roll and pitch damping at `WAIST_KD`
(the G1's is 2 x DAMPING_5020, about 1.81), everything else the G1's.

On the real G1 (2026-10-27) the deploy-side waist roll/pitch kd of 3.0 was tried against
the three-robot policy's chatter; this variant trains with it. The model, keyframe,
collisions, translator, tracked bodies and action scale (from kp, unchanged) are the
G1's, so the G1's reference motions and subsets serve unchanged.
"""

from __future__ import annotations

import dataclasses

from morphmimic.robots.spec import RobotSpec

if TYPE_CHECKING := False:  # pragma: no cover
  from mjlab.entity import EntityCfg

WAIST_KD = 3.0


def get_g1_amber_robot_cfg() -> EntityCfg:
  """A fresh G1 entity cfg whose waist roll/pitch actuator has damping `WAIST_KD`."""
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTUATOR_WAIST,
    get_g1_robot_cfg,
  )

  base = get_g1_robot_cfg()
  actuators = tuple(
    dataclasses.replace(a, damping=WAIST_KD)
    if a.target_names_expr == G1_ACTUATOR_WAIST.target_names_expr
    else a
    for a in base.articulation.actuators
  )
  assert actuators != base.articulation.actuators, "no G1 waist actuator to change"
  articulation = dataclasses.replace(base.articulation, actuators=actuators)
  return dataclasses.replace(base, articulation=articulation)


def g1_amber_spec() -> RobotSpec:
  from morphmimic.robots.g1 import g1_spec

  return dataclasses.replace(g1_spec(), name="g1_amber", entity_cfg_fn=get_g1_amber_robot_cfg)
