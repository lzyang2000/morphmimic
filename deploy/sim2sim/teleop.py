"""The policy node's live reference: GMR streamer frames -> the policy's K-frame windows.

`TeleopReference` replaces the clip (`policy_node.py --source teleop`): it subscribes to
the GMR streamer (`deploy/vr_teleop/streamer.py --robot ROBOT`, `RefMessage` chunks of
reference states [240]) and, once per control step, returns the window of K reference
states the policy tracks. The mode logic is SONIC's, shared with the five-point sim
(`vr_teleop.controller.SimController`): follow the streamer's mode, IDLE after
`TIMEOUT_S` of silence, a jitter buffer `DELAY` frames deep, OFF latched after a fall
(the robot reset) until a recalibration.

What each mode tracks:
- POSE: the operator's frames, turned at POSE entry so the operator's pelvis heading
  is the robot's (`turn_states`), then that same turn until the next entry; blended
  over `BLEND_S` from the window tracked before.
- IDLE and OFF: the neutral stand (`stand.json`, the robot's retarget of one SEED
  standing frame; the G1's with its ankle offset) at the robot's heading on entry, blended
  over `STAND_BLEND_S`, in the sim and on the robot alike: OFF does not damp (the remote's
  B does).
- POSE_PAUSE: the window held when it began.

Only the heading is aligned, never the position: the goals are position-free (the
motion goal's anchor path is in the reference's own heading frame, the five-point
positions in the reference pelvis's), so the robot needs no odometry. The orientations
are relative to the robot's own (IMU and joint encoders, `Robot.slot0_quat`).

`HumanTeleopReference` is the same for the human interface (`policy_node.py --interface
human`): it reads the human streamer (`streamer.py --target human`, frames of the
operator on the uniform skeleton) and returns human windows [K, 168]
(`deploy/sim2sim/human.py`), the stand being the uniform skeleton's (`human_stand`).
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import zmq

from deploy.sim2sim.human import blend_human, human_heading, human_stand, turn_human
from deploy.sim2sim.obs import GOAL_K, Robot, quat_mul, slerp
from deploy.vr_teleop.constants import DELAY, FPS, STAND_BLEND_S, TIMEOUT_S
from deploy.vr_teleop.controller import SimController, recovery_steps
from deploy.vr_teleop.controls import POSE, POSE_PAUSE
from deploy.vr_teleop.gmr_retarget import STAND_FILE
from deploy.vr_teleop.protocol import decode_ref

NUM_BODIES = 14
_POS, _QUAT = slice(58, 100), slice(100, 156)
_LIN, _ANG = slice(156, 198), slice(198, 240)


def heading(quat: np.ndarray) -> float:
  """Yaw (rad) of a wxyz quaternion: the xy angle of its x axis."""
  w, x, y, z = quat
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _bodies(states: np.ndarray, part: slice, width: int) -> np.ndarray:
  return states[..., part].reshape(*states.shape[:-1], NUM_BODIES, width)


def turn_states(states: np.ndarray, yaw: float, origin: np.ndarray) -> np.ndarray:
  """Reference states [..., 240] turned by `yaw` about the vertical through `origin`
  (x, y): body positions, orientations and velocities; joints unchanged."""
  c, s = math.cos(yaw), math.sin(yaw)
  rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  out = states.copy()
  pos = _bodies(states, _POS, 3) - np.array([origin[0], origin[1], 0.0])
  out[..., _POS] = (pos @ rot.T).reshape(*states.shape[:-1], -1)
  turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  out[..., _QUAT] = quat_mul(turn, _bodies(states, _QUAT, 4)).reshape(*states.shape[:-1], -1)
  for part in (_LIN, _ANG):
    out[..., part] = (_bodies(states, part, 3) @ rot.T).reshape(*states.shape[:-1], -1)
  return out


def blend_states(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
  """`a` -> `b` [..., 240] by `alpha`: linear, the body orientations slerped."""
  out = (1.0 - alpha) * a + alpha * b
  q = slerp(_bodies(a, _QUAT, 4), _bodies(b, _QUAT, 4), alpha)
  out[..., _QUAT] = q.reshape(*a.shape[:-1], -1)
  return out


class TeleopReference:
  """The live reference windows of one robot (see the module docstring). The window
  layout is a robot reference state per frame; `HumanTeleopReference` overrides `kind`
  and the frame operations below for human frames."""

  kind = "ref"  # The stream messages it reads (`protocol.RefMessage.kind`).

  def __init__(self, robot: Robot, endpoint: str, delay: int = DELAY,
               timeout_s: float = TIMEOUT_S, clock=time.monotonic) -> None:
    self.robot = robot
    self.controller = SimController(delay=delay, timeout_s=timeout_s)
    self.clock = clock
    self.stand = self._stand()
    self._sock = zmq.Context.instance().socket(zmq.SUB)
    self._sock.setsockopt(zmq.SUBSCRIBE, b"")
    self._sock.setsockopt(zmq.LINGER, 0)
    self._sock.connect(endpoint)
    self.endpoint = endpoint
    self.bad_messages = 0
    self._turn: tuple[float, np.ndarray] | None = None  # The POSE anchor (yaw, origin).
    self._last: np.ndarray | None = None  # The window returned last step.
    self._blend_from: np.ndarray | None = None
    self._stand_to: np.ndarray | None = None
    self._stand_steps = 0

  def close(self) -> None:
    self._sock.close()

  def drain(self) -> list:
    msgs = []
    while True:
      try:
        raw = self._sock.recv(zmq.NOBLOCK)
      except zmq.Again:
        return msgs
      try:
        msg = decode_ref(raw)
      except Exception as e:  # noqa: BLE001 - a bad message is counted and skipped.
        self.bad_messages += 1
        if self.bad_messages == 1:
          print(f"[policy_node] skipping a bad stream message: {e}", flush=True)
        continue
      self._check(msg)
      msgs.append(msg)

  def reset(self) -> None:
    """The robot was reset (it fell): hold OFF until the operator recalibrates."""
    self.controller.latch_off()
    print(f"[policy_node] robot reset: OFF; {recovery_steps(self.controller.stream_mode)}",
          flush=True)

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """This control step's window (K reference states [K, 240]; human frames [K, 168]
    in `HumanTeleopReference`), from the robot's IMU orientation and native joint
    positions (`step_id` is unused: the stream's own clock paces the frames)."""
    c = self.controller
    previous = c.mode
    mode = c.tick(self.clock(), self.drain())
    if mode != previous:
      print(f"[policy_node] mode {mode} (calib_id {c.calib_id})", flush=True)
    robot_yaw = heading(self.robot.slot0_quat(imu_quat, q))
    if mode == POSE:
      win = self._pose(robot_yaw)
    elif mode == POSE_PAUSE and self._last is not None:
      win = self._last
    else:  # OFF, IDLE (and a pause before anything was tracked).
      win = self._standing(robot_yaw, entered=mode != previous)
    self._last = win
    return win

  def _pose(self, robot_yaw: float) -> np.ndarray:
    c = self.controller
    frames = c.timeline.frames()
    if frames is None:  # Nothing streamed yet: keep what was tracked.
      return self._last if self._last is not None else self._standing(robot_yaw, True)
    live = np.stack([f.state for f in frames])
    if c.entered_pose or c.calib_changed or self._turn is None:
      origin = self._origin(live[0]).copy()
      self._turn = (robot_yaw - self._heading(live[0]), origin)
      self._blend_from = self._last
    live = self._turned(live, *self._turn)  # About the entry pelvis, which goes to x = y = 0.
    if c.blend_alpha < 1.0 and self._blend_from is not None:
      live = self._blend(self._blend_from, live, c.blend_alpha)
    return live

  def _standing(self, robot_yaw: float, entered: bool) -> np.ndarray:
    if entered or self._stand_to is None:
      self._stand_to = np.repeat(self._turned(self.stand, robot_yaw, np.zeros(2))[None],
                                 GOAL_K, 0)
      self._blend_from = self._last if self._last is not None else self._stand_to
      self._stand_steps = 0
      self._turn = None
    else:
      self._stand_steps += 1
    alpha = min(1.0, self._stand_steps / max(1, round(STAND_BLEND_S * FPS)))
    return self._blend(self._blend_from, self._stand_to, alpha)

  # The frame layout: a robot reference state [240].

  def _check(self, msg) -> None:
    """Refuses (ValueError) a stream of the other kind or for another robot."""
    if msg.kind != "ref":
      raise ValueError("the streamer sends the human skeleton (--target human): run the"
                       " policy node with --interface human, or restart the streamer with"
                       f" --robot {self.robot.body}")
    if msg.robot != self.robot.body:
      raise ValueError(f"the streamer retargets to {msg.robot}, this policy node runs "
                       f"{self.robot.name}: restart the streamer with --robot {self.robot.body}")

  def _stand(self) -> np.ndarray:
    return np.asarray(json.loads(STAND_FILE.read_text())[self.robot.body]["state"])

  @staticmethod
  def _origin(frame: np.ndarray) -> np.ndarray:
    return frame[58:60]

  @staticmethod
  def _heading(frame: np.ndarray) -> float:
    return heading(frame[100:104])

  @staticmethod
  def _turned(frames: np.ndarray, yaw: float, origin: np.ndarray) -> np.ndarray:
    return turn_states(frames, yaw, origin)

  @staticmethod
  def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return blend_states(a, b, alpha)


class HumanTeleopReference(TeleopReference):
  """The live human windows [K, 168] (`policy_node.py --interface human --source
  teleop`): the same modes, jitter buffer and heading alignment, on frames of the
  operator on the uniform skeleton; the heading is the human pelvis's in the robot's
  convention (`human_heading`), the stand the uniform skeleton's."""

  kind = "human"

  def _check(self, msg) -> None:
    if msg.kind != "human":
      raise ValueError(f"the streamer retargets to {msg.robot} (GMR): this policy node reads"
                       " the human skeleton; restart the streamer with --target human")

  def _stand(self) -> np.ndarray:
    return human_stand()

  @staticmethod
  def _origin(frame: np.ndarray) -> np.ndarray:
    return frame[:2]  # The Pelvis slot's position comes first.

  @staticmethod
  def _heading(frame: np.ndarray) -> float:
    return human_heading(frame)

  @staticmethod
  def _turned(frames: np.ndarray, yaw: float, origin: np.ndarray) -> np.ndarray:
    return turn_human(frames, yaw, origin)

  @staticmethod
  def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return blend_human(a, b, alpha)
