"""Where the robot's pose comes from, for the robot-anchored motion goal
(`--goal-anchor robot`): the goal is the reference relative to the robot's own anchor,
so the policy node needs the robot's position and heading in the reference's world,
which encoders and an IMU do not give.

A `PoseSource` gives the floating base's (the robot's root body: the G1 and X2
pelvis, the T2 trunk) world position and yaw. The robot side reads it each control step
and sends it in the state message (`protocol.pack_state(pelvis_pos=, pelvis_yaw=)`);
the policy node turns it into the anchor pose (`obs.Robot.anchor_pose`). A robot
without a source sends NaN, and a robot-anchor policy refuses to run on it.

- `SimPoseSource`: the simulator's true pose (`sim_node.py`).
- The real robot's source is a state estimator, the SE(3) LIO
  (https://github.com/lzyang2000/se3-lio), a later task: it implements `pose()` from
  the estimator's latest base pose, and `deploy/hardware/g1_node.py` passes it to
  `pack_state` where it now sends NaN (and stops refusing robot-anchor exports).

numpy only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# The policy node's refusal when a robot-anchor export meets a robot without a pose.
NO_POSE_SOURCE = (
  "--goal-anchor robot needs a pose source (state estimator); none on this robot"
)


@runtime_checkable
class PoseSource(Protocol):
  def pose(self) -> tuple[np.ndarray, float]:
    """(world position [3] (m), yaw (rad)) of the floating base, now."""
    ...


class SimPoseSource:
  """The simulator's floating-base pose: the free joint's body, read from MuJoCo's
  kinematics (`data.xpos` / `data.xquat`, the same instant the simulated IMU reads)."""

  def __init__(self, model, data) -> None:
    free = np.flatnonzero(np.asarray(model.jnt_type) == 0)  # mjJNT_FREE
    if len(free) != 1:
      raise ValueError(f"expected one free joint, found {len(free)}")
    self.data = data
    self.body = int(model.jnt_bodyid[free[0]])

  def pose(self) -> tuple[np.ndarray, float]:
    w, x, y, z = self.data.xquat[self.body]
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return self.data.xpos[self.body].copy(), yaw
