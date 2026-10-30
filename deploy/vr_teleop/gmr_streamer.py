"""PICO body -> GMR -> the robot's reference states, for the sim2sim policy node.

`GmrTarget` is the streamer's target (`streamer.py --robot ROBOT`) for any of our robots:
each 50 Hz frame of the operator's PICO body goes through GMR (`gmr_retarget.py`: the
arm abduction map and the IK) and comes out as one reference state `[240]`, the layout
of the sim2sim clips (`deploy/sim2sim/obs.py`): the canonical joints (the robot's
translator) and their velocities, then the 14 tracked bodies' positions, orientations
and velocities, read from GMR's own MuJoCo model after the solve (the same kinematics as
the training models). Velocities are backward differences at 50 Hz.

PICO frames: GMR's configs (its G1 one, our fitted X2 and T2 ones) are authored in the
frame GMR's `XRobotStreamer` makes of the raw samples: positions `M p`, orientations
`q_M * q` with `M` = [[1, 0, 0], [0, 0, -1], [0, 1, 0]] (Unity's y-up to z-up). That is
not SONIC's transform (`retarget.unity_to_robot`, the five-point path), so this module
does GMR's (`pico_to_gmr`); `gmr_to_pico` inverts it for synthetic sessions.

Calibration (A+B+X+Y, the operator standing): GMR solves the pose to convergence from
the neutral stand (`GmrRetargeter.settle`) and a height offset puts the robot's ankles
at the stand's height (`stand.json`), following the floor after that
(`floor.FloorTracker`, for slopes, stairs and PICO's drift); the policy goals need no
other placement, since they are position-free and the policy node aligns the heading.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.sim2sim.human import PICO_JOINTS, SMPL_PARENT
from deploy.sim2sim.obs import Robot
from deploy.vr_teleop.floor import FloorTracker
from deploy.vr_teleop.gmr_retarget import STAND_FILE, GmrRetargeter
from deploy.vr_teleop.protocol import RefFrame, RefMessage, encode_ref

FPS = 50
_M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_M_ROT = R.from_matrix(_M)

Body = dict[str, tuple[np.ndarray, np.ndarray]]


def pico_to_gmr(body: np.ndarray) -> Body:
  """PICO's `[24, 7]` (Unity frame, xyz + quaternion xyzw) -> GMR's human frame
  `{joint: (pos [3], quat wxyz)}` (`XRobotStreamer.get_processed_body_data`)."""
  pos = body[:, :3] @ _M.T
  quat = (_M_ROT * R.from_quat(body[:, 3:])).as_quat(scalar_first=True)
  return {name: (pos[i], quat[i]) for i, name in enumerate(PICO_JOINTS)}


def gmr_to_pico(frame: Body) -> np.ndarray:
  """GMR's human frame -> PICO's `[24, 7]` (the inverse of `pico_to_gmr`). A joint the
  frame lacks takes its nearest SMPL ancestor's pose."""
  out = np.zeros((24, 7))
  for i in range(24):
    j = i
    while PICO_JOINTS[j] not in frame:
      j = SMPL_PARENT[j]
    pos, quat = frame[PICO_JOINTS[j]]
    out[i, :3] = _M.T @ np.asarray(pos)
    out[i, 3:] = (_M_ROT.inv() * R.from_quat(quat, scalar_first=True)).as_quat()
  return out


class ReferenceBuilder:
  """One robot's reference states from GMR human frames, 50 Hz apart."""

  def __init__(self, robot: str, human_height: float | None = None) -> None:
    self.robot = Robot.load(robot)
    self.gmr = GmrRetargeter(robot, human_height)
    m = self.gmr.gmr.model
    self._bodies = [m.body(b).id for b in self.robot.tracked_bodies]
    self._ankles = [self.robot.tracked_bodies.index(f"{s}_ankle_roll_link")
                    for s in ("left", "right")]
    self._stand = json.loads(STAND_FILE.read_text())[self.robot.body]
    self.z_offset = 0.0
    self.floor = FloorTracker()
    self._prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

  def calibrate(self, body: Body) -> None:
    """Settle GMR on the calibration pose and put the robot's ankles at the stand's
    height."""
    self.gmr.settle(body)
    data = self.gmr.gmr.configuration.data
    self.floor.reset(data.xpos[[self._bodies[i] for i in self._ankles]])
    self.z_offset = self._stand["ankle_z"] - self.floor.z
    self._prev = None

  def state(self, body: Body) -> np.ndarray:
    """The reference state [240] of one frame (after `calibrate`)."""
    pose = self.gmr.retarget(body)
    stand = self._stand["joint_pos"]
    q = np.array([pose.joint_pos.get(n, stand[n]) for n in self.robot.native_joint_names])
    data = self.gmr.gmr.configuration.data
    pos = data.xpos[self._bodies].copy()
    self.z_offset = self._stand["ankle_z"] - self.floor.update(pos[self._ankles])
    pos[:, 2] += self.z_offset
    quat = data.xquat[self._bodies].copy()
    q_canon = self.robot.to_canonical(q)
    if self._prev is None:
      qd, lin, ang = np.zeros(len(q_canon)), np.zeros_like(pos), np.zeros_like(pos)
    else:
      q0, pos0, quat0 = self._prev
      qd = (q_canon - q0) * FPS
      lin = (pos - pos0) * FPS
      turn = R.from_quat(quat, scalar_first=True) * R.from_quat(quat0, scalar_first=True).inv()
      ang = turn.as_rotvec() * FPS  # World frame, as the dataset's.
    self._prev = (q_canon, pos, quat)
    return np.concatenate((q_canon, qd, pos.ravel(), quat.ravel(), lin.ravel(), ang.ravel()))


class GmrTarget:
  """The streamer target sending `robot`'s reference states (`RefMessage`)."""

  def __init__(self, robot: str, human_height: float | None = None) -> None:
    self.name = robot
    self.builder = ReferenceBuilder(robot, human_height)
    self.calibrated = False

  def calibrate(self, body: np.ndarray) -> None:
    self.builder.calibrate(pico_to_gmr(body))
    self.calibrated = True
    print(f"[streamer] GMR {self.name}: calibrated, height offset "
          f"{self.builder.z_offset:+.3f} m", flush=True)

  def frame(self, index: int, body: np.ndarray) -> RefFrame:
    return RefFrame(index, self.builder.state(pico_to_gmr(body)))

  def message(self, mode: str, calib_id: int, t_ns: int, fps: float,
              frames: list) -> tuple[RefMessage, bytes]:
    msg = RefMessage(mode=mode, calib_id=calib_id, t_ns=t_ns, fps=fps, robot=self.name,
                     frames=frames)
    return msg, encode_ref(msg)
