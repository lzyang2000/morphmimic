"""The robot side of the sim2sim protocol on hardware, for any robot: the button state
machine, the 50 Hz policy loop and the safety layer, over the three things a robot has to
provide (`G1Node` in `g1_node.py`).

A subclass implements `read_state()` (the latest `Sample`: the trained IMU body's
orientation and gyro, the joint positions and velocities, no waiting) and `send(q_target,
kp, kd)` (one PD command). It can override `buttons_held()` (the button names held now,
the union of the `buttons` sources' `held()` by default: a pad, the terminal keys),
`idle_hold()` (what the node does each tick while it waits for START: damping at the
current pose), `may_stand()` (False refuses a START press; the node waits for the next;
the base refuses a state `stale()` objects to),
`record_extra()` (more columns for the `Recording`'s rows), `faults(sample)` (None, or
the robot's own fault, which damps) and `stale(sample)` (None, or the reason this state
must not reach the policy: the base's `invalid state`, then the robot's own, such as
stale inputs).

Safety, the same on every robot. Each tick of every state that commands the robot (the
policy loop and its PD stand on a stalled policy or bad data, the stand hold, the 2 s
ramps to the stand) reads the buttons and the state and acts on the first of:

  1. the operator: SELECT damps and exits, B damps
  2. damp (`Tripped`): `tilt`, the pelvis's up axis more than `tilt_deg` (80) from
     vertical while the policy or a blend to the stand drives the robot, more than
     `stand_tilt_deg` (30) once the target is the stand itself (the stand hold, a PD
     stand whose blend has arrived), computed from a valid state only; then the robot's
     own `faults()`; in a PD stand as anywhere
  3. the PD stand: an invalid state (a non-finite `quat`, `gyro`, `q` or `dq`, or a
     quaternion of norm below 0.5), stale data where the robot reports ages, a stalled
     policy; the policy resumes on the stand once it clears. Only the policy loop acts
     on it: the stand hold reports it once, and the ramps head for the stand anyway.
  4. the policy

The state machine takes five button names, from any source:

  start   from damping, interpolate over `STAND_S` to the stand and hold it
  A       start the policy loop (the step count restarts at 0)
  Y       during the policy loop: a cue (counted once per press)
  B       graceful stop: damp (kp 0, kd only) at the current pose; START stands again
  select  emergency stop: damp and exit

The loop sends each step's state message and applies the action of the same `step_id`.
It waits up to `action_wait_s` for it (else it holds the last target), and if no new
action arrives for `action_timeout_s` (the policy node stalled or died) it restarts the
step count, so the policy node resets onto the stand (a clip is dropped), and blends from
the last target to the PD stand until the policy answers; its first answer takes over
again, on the stand (B damps). A state `stale()` objects to takes the same path at once:
no state goes out, and the first usable one goes out as step 0. An action that is not
all finite counts as late. With `max_steps` N the policy runs N steps (50 per
second), then the node blends over 2 s back to the PD stand and holds it: A runs the
policy again, B damps. A trip damps like B and waits for START.

Its logic runs against any subclass, so it can be tested with a fake robot.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq

from deploy.sim2sim import protocol
from deploy.sim2sim.obs import Robot, quat_apply_inverse

STAND_FILE = Path(__file__).resolve().parents[1] / "vr_teleop/gmr_configs/stand.json"
STAND_S = 2.0  # START: interpolation time to the stand.
DAMP_TICKS = 50  # Damping commands sent on a stop (1 s).
STAND, GO, DAMP, KILL, CUE = "start", "A", "B", "select", "Y"
DOWN = np.array([0.0, 0.0, -1.0])


class Exit(Exception):
  """SELECT: damp and leave."""


class Tripped(Exception):
  """Damp now (a fall, the robot's own fault, or B outside the policy loop): `str(e)` is
  the reason."""


@dataclass
class Gains:
  kp: np.ndarray
  kd: np.ndarray

  @classmethod
  def from_deploy(cls, robot_doc: dict, kp_scale: float = 1.0) -> Gains:
    pd = robot_doc["pd"]
    return cls(np.asarray(pd["kp"], float) * kp_scale, np.asarray(pd["kd"], float))


@dataclass
class Sample:
  """One reading of the robot, in the policy's frame."""

  quat: np.ndarray  # wxyz, the trained IMU body's orientation
  gyro: np.ndarray  # body frame
  q: np.ndarray
  dq: np.ndarray


def valid(sample: Sample) -> bool:
  """All finite, and a quaternion that is not near zero (an uninitialised IMU)."""
  finite = all(np.isfinite(x).all() for x in (sample.q, sample.dq, sample.quat, sample.gyro))
  return finite and np.linalg.norm(sample.quat) >= 0.5


@dataclass
class Recording:
  rows: dict = field(default_factory=lambda: {k: [] for k in (
    "t", "t_state", "t_sent", "step_id", "quat", "gyro", "q", "dq", "target", "fresh")})

  def add(self, **row) -> None:
    for k, v in row.items():
      self.rows.setdefault(k, []).append(np.asarray(v))

  def save(self, path: Path) -> int:
    np.savez_compressed(path, **{k: np.array(v) for k, v in self.rows.items()})
    return len(self.rows["t"])


class HardwareNode(ABC):
  """The button state machine and the 50 Hz policy loop over a robot's `read_state` and
  `send`."""

  tag: str = "node"  # The print prefix: `[<tag>]`.

  def __init__(self, robot: Robot, gains: Gains, stand_q: np.ndarray,
               buttons: Sequence = (), tilt_deg: float = 80.0, stand_tilt_deg: float = 30.0,
               policy_host: str = protocol.HOST, bind_host: str = protocol.HOST,
               action_wait_s: float = 0.010, action_timeout_s: float = 1.0,
               send_delay_s: float = 0.0, max_steps: int | None = None, clock=time.perf_counter, sleep=time.sleep,
               record: Recording | None = None, state_port: int = protocol.STATE_PORT,
               action_port: int = protocol.ACTION_PORT):
    self.robot, self.gains = robot, gains
    self.stand_q = np.asarray(stand_q, float)
    self.n = robot.num_native
    self.dt = robot.control_dt
    self.buttons = buttons
    self.tilt_deg, self.stand_tilt_deg = tilt_deg, stand_tilt_deg
    self.action_wait_s, self.action_timeout_s = action_wait_s, action_timeout_s
    # Extra hold of each fresh policy command before it goes out: emulates a slower policy
    # step (e.g. the Jetson's) on a faster host.
    self.send_delay_s = send_delay_s
    self.max_steps = max_steps
    self.target = self.stand_q.copy()  # The last policy target applied.
    self.clock, self.sleep = clock, sleep
    self.record = record
    self.action_in = protocol.receiver(action_port, bind_host, timeout=0.0)
    self.state_out = protocol.sender(state_port, policy_host, timeout=0.002)

  def close(self) -> None:
    self.action_in.close()
    self.state_out.close()

  # What a robot provides.

  @abstractmethod
  def read_state(self) -> Sample:
    """The latest reading, without waiting."""

  @abstractmethod
  def send(self, q_target: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
    """One PD command, in the native joint order."""

  def buttons_held(self) -> set[str]:
    """The button names held now, from every source. Called once per tick."""
    return set().union(*(source.held() for source in self.buttons))

  def idle_hold(self) -> None:
    """Each tick while waiting for START: damping at the current pose."""
    self.send(self._limp_q(), np.zeros(self.n), self.gains.kd)

  def may_stand(self) -> bool:
    """After a START press, before the ramp: False damps until START is released, then
    waits for the next press. The base refuses a state `stale()` objects to (the ramp
    starts from the pose it reads right after)."""
    reason = self.stale(self.read_state())
    if reason is not None:
      print(f"[{self.tag}] no START: {reason}", flush=True)
    return reason is None

  def record_extra(self) -> dict:
    """More columns for each policy tick's `Recording` row."""
    return {}

  def faults(self, sample: Sample) -> str | None:
    """None, or the robot's own fault that damps (the base knows none)."""
    return None

  def stale(self, sample: Sample) -> str | None:
    """None, or the reason `sample` must not reach the policy (the policy loop goes to the
    PD stand until it is usable again, as for a stalled policy node): `invalid state`; a
    robot adds its own reasons after it."""
    return None if valid(sample) else "invalid state"

  def _q(self) -> np.ndarray:
    return self.read_state().q

  def _limp_q(self) -> np.ndarray:
    """The measured pose as a kp-0 target, a non-finite position as 0.0: with kp 0 the
    position does not act, but a driver computing kp * (target - q) gets NaN from a NaN."""
    return np.nan_to_num(self._q(), nan=0.0, posinf=0.0, neginf=0.0)

  def _buttons(self, *names: str) -> str | None:
    held = self.buttons_held()
    return next((n for n in names if n in held), None)

  def _check(self, sample: Sample, limit_deg: float) -> None:
    """The damp checks: `tilt`, the pelvis's up axis more than `limit_deg` from vertical
    (never from an invalid state: a bad quaternion is `stale()`'s), then the robot's
    `faults()`, which do not depend on the state's validity."""
    if valid(sample):
      up = -quat_apply_inverse(sample.quat / np.linalg.norm(sample.quat), DOWN)[2]
      deg = np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))
      if deg > limit_deg:
        raise Tripped(f"tilt {deg:.0f} deg > {limit_deg:g}")
    reason = self.faults(sample)
    if reason is not None:
      raise Tripped(reason)

  # States.

  def damp(self, ticks: int = DAMP_TICKS) -> None:
    """Zero stiffness, damping only, at the current pose."""
    for _ in range(ticks):
      self.send(self._limp_q(), np.zeros(self.n), self.gains.kd)
      self.sleep(self.dt)

  def wait_for(self, *buttons: str, hold) -> str:
    """Hold (call `hold` every tick) until one of `buttons`, which it returns, the first
    of them if several are held; SELECT exits, before any of them."""
    while True:
      hit = self._buttons(KILL, *buttons)
      if hit == KILL:
        raise Exit
      if hit is not None:
        return hit
      hold()
      self.sleep(self.dt)

  def to_stand(self, start: np.ndarray) -> None:
    """PD targets from `start` to the stand over `STAND_S`, the damp checks each tick (a
    robot that falls in the blend back after the policy is damped at once); B damps."""
    steps = max(1, round(STAND_S / self.dt))
    for i in range(steps):
      hit = self._buttons(KILL, DAMP)
      if hit == KILL:
        raise Exit
      if hit == DAMP:
        raise Tripped(DAMP)
      self._check(self.read_state(), self.tilt_deg)
      alpha = (i + 1) / steps
      self.send((1 - alpha) * start + alpha * self.stand_q, self.gains.kp, self.gains.kd)
      self.sleep(self.dt)

  def hold_stand(self) -> None:
    """Hold the stand by PD until A; B damps. The damp checks each tick; a state `stale()`
    objects to is reported once and the hold goes on (the stand needs no data)."""
    told = False

    def hold() -> None:
      nonlocal told
      sample = self.read_state()
      self._check(sample, self.stand_tilt_deg)
      reason = self.stale(sample)
      if reason is not None and not told:
        print(f"[{self.tag}] {reason} (holding the stand)", flush=True)
        told = True
      self.send(self.stand_q, self.gains.kp, self.gains.kd)

    if self.wait_for(DAMP, GO, hold=hold) == DAMP:
      raise Tripped(DAMP)

  def stand_up(self) -> None:
    print(f"[{self.tag}] press {STAND} (remote or pad) to stand ({STAND_S:.0f} s to the stand)",
          flush=True)
    while True:
      self.wait_for(STAND, hold=self.idle_hold)
      if self.may_stand():
        break
      # Refused: keep damping until START is released, so a held press is one refusal.
      while (hit := self._buttons(KILL, STAND)) is not None:
        if hit == KILL:
          raise Exit
        self.idle_hold()
        self.sleep(self.dt)
    self.to_stand(self._q())
    print(f"[{self.tag}] standing; press {GO} to start the policy, {DAMP} to damp", flush=True)
    self.hold_stand()

  def policy_loop(self) -> str:
    """The 50 Hz loop until B ('damp'), `max_steps` ('done') or SELECT (Exit).

    No action for `action_timeout_s` (a stalled or dead policy node): the step count and
    the cues restart at 0, which the policy node takes as a reset (it clears the clip and
    tracks the stand, facing the robot), and until it answers again the targets blend
    from the last one to the PD stand over `STAND_S` and hold it. Its first answer takes
    over again, on the stand. A state `stale()` objects to (stale or invalid data) does
    not go out and enters the same path at once; while it lasts the step count and the
    cues stay at their reset, so the first usable state goes out as step 0."""
    print(f"[{self.tag}] policy running"
          + (f" for {self.max_steps} steps" if self.max_steps is not None else ""), flush=True)
    step, target, last_fresh = 0, self.stand_q.copy(), self.clock()
    misses, cues, cue_held = 0, 0, False
    stalled: list | None = None  # [the target it stalled at, ticks since]
    blend_ticks = max(1, round(STAND_S / self.dt))
    next_tick = self.clock()
    while self.max_steps is None or step < self.max_steps:
      delay = next_tick - self.clock()
      if delay > 0:
        self.sleep(delay)
      next_tick += self.dt
      hit = self._buttons(KILL, DAMP, CUE)
      if hit == KILL:
        raise Exit
      if hit == DAMP:
        return "damp"
      if hit == CUE and not cue_held and stalled is None:  # A press, not a hold.
        cues += 1
        print(f"[{self.tag}] {CUE}: cue {cues}", flush=True)
      cue_held = hit == CUE
      sample = self.read_state()
      t_state = self.clock()
      # The stand's limit once the target is the stand itself: the fallback blend arrived.
      self._check(sample, self.stand_tilt_deg if stalled is not None and stalled[1] >= blend_ticks
                  else self.tilt_deg)
      reason = self.stale(sample)
      sent = step
      if reason is not None:  # Nothing goes out, nothing is awaited.
        fresh = None
      else:
        try:  # At most 2 ms on a missing policy node (then: a late action).
          # No pose source yet: the pose goes as NaN (the SE(3) LIO plugs in here).
          self.state_out.send(protocol.pack_state(step, sample.quat, sample.gyro, sample.q,
                                                  sample.dq, cues))
          fresh = self._await_action(step)
        except zmq.Again:
          fresh = None
      if fresh is not None:
        if stalled is not None:
          print(f"[{self.tag}] the policy answers again: on the stand (step {step})", flush=True)
          stalled = None
        target, last_fresh = fresh, self.clock()
      elif stalled is not None:
        stalled[1] += 1
        alpha = min(1.0, stalled[1] / blend_ticks)
        target = (1 - alpha) * stalled[0] + alpha * self.stand_q
      elif reason is not None:
        print(f"[{self.tag}] {reason}: to the PD stand until the data is back, then the policy"
              f" on the stand; {DAMP} damps", flush=True)
        stalled = [target.copy(), 0]
      else:
        misses += 1
        if self.clock() - last_fresh > self.action_timeout_s:
          print(f"[{self.tag}] no action for {self.action_timeout_s:.2f} s (policy node down?):"
                f" to the PD stand until it answers, then the policy on the stand; {DAMP}"
                " damps", flush=True)
          stalled, step, cues = [target.copy(), 0], -1, 0  # -1: the next step is 0.
      if reason is not None:  # Every stale tick, so the first usable state goes out as 0.
        step, cues = -1, 0
      if fresh is not None and self.send_delay_s > 0:
        self.sleep(max(0.0, t_state + self.send_delay_s - self.clock()))
      self.send(target, self.gains.kp, self.gains.kd)
      t_sent = self.clock()
      self.target = target
      if self.record is not None:
        self.record.add(t=self.clock(), t_state=t_state, t_sent=t_sent, step_id=sent, quat=sample.quat, gyro=sample.gyro,
                        q=sample.q, dq=sample.dq, target=target, fresh=fresh is not None,
                        **self.record_extra())
      step += 1
      if reason is None and step % 500 == 0 and misses:  # Stale ticks all leave step at 0.
        print(f"[{self.tag}] step {step}: {misses} late actions so far (held the last)",
              flush=True)
    print(f"[{self.tag}] {step} policy steps done ({misses} late): back to the stand",
          flush=True)
    return "done"

  def _await_action(self, step: int) -> np.ndarray | None:
    """The action for `step`, waiting up to `action_wait_s`; None if it is late or not all
    finite."""
    deadline = self.clock() + self.action_wait_s
    while True:
      remaining = deadline - self.clock()
      self.action_in.setsockopt(zmq.RCVTIMEO, max(0, round(remaining * 1e3)))
      try:
        got, targets = protocol.unpack_action(self.action_in.recv(), self.n)
      except zmq.Again:
        return None
      if got == step:
        return targets if np.isfinite(targets).all() else None
      if remaining <= 0:
        return None

  def run(self) -> None:
    """Stand, run, damp, repeat until SELECT or Ctrl-C; always ends damped. The policy
    loop's end ('done') or B: back to the PD stand, or damp. A trip damps like B."""
    try:
      while True:
        try:
          self.stand_up()
          while self.policy_loop() != "damp":
            self.to_stand(self.target)
            print(f"[{self.tag}] holding the stand; {GO} runs the policy again, {DAMP} damps",
                  flush=True)
            self.hold_stand()
        except Tripped as tripped:
          print(f"[{self.tag}] {tripped}: damping", flush=True)
        self.damp()
        print(f"[{self.tag}] damped; {STAND} stands again, {KILL} or Ctrl-C exits", flush=True)
    except Exit:
      print(f"[{self.tag}] {KILL}: emergency stop", flush=True)
    except KeyboardInterrupt:
      pass
    finally:
      self.damp()


def stand_pose(robot: Robot, which: str) -> np.ndarray:
  """The pose START stands the robot in, in native joint order."""
  if which == "default":
    return robot.default_joint_pos.copy()
  stand = json.loads(STAND_FILE.read_text())[robot.body]["joint_pos"]
  return np.array([stand[n] for n in robot.native_joint_names])
