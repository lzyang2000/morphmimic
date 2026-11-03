"""The robot-side policy process: numpy, onnx and onnxruntime only (no torch, no mjlab).

Each `protocol` state message (joint encoders and the IMU) is answered with joint
targets. Per message it builds the policy inputs from the measurements and the
reference motion (`obs.py`), runs the deployment folder's `policy.onnx` with the
robot's inputs from `<robot>.json`, and turns its motion action (`action_motion`) into
native joint targets. A `step_id` lower than the previous one restarts the reference and
clears the history.

The reference is a recorded clip (`--source clip`: `export_assets.py`'s `<clip>.npz`, its
canonical states, played from frame 0 at each reset; with `--from-stand S`, the robot's
case on hardware, the robot stands until a cue (the remote's Y), then the clip, turned to
the robot's heading, is led into from the stand over S seconds, `CuedClipReference`) or the operator, live
(`--source teleop`: the GMR streamer's reference states, through SONIC's modes,
`teleop.TeleopReference`), or the neutral stand held still (`--source stand`,
`StandReference`, the default without `--clip`: a first run on hardware). In teleop, stand
and a clip from the stand the window the policy tracks is also pushed to
`protocol.DISPLAY_PORT`, where the sim draws it.

The policy acts through the motion goal, in the export's goal anchor (`<robot>.json`
`goal_anchor`; "reference" when absent, every export before the switch). Under "robot"
(`--goal-anchor robot`, goal version 1's robot-relative goal) the goal needs the robot's
pose: the state message's floating-base position and yaw (the sim node sends the
simulator's; on hardware a state estimator, `deploy/hardware/pose_source.py`), and the
node refuses to act on a message without one; it also refuses `--source teleop`,
`--source stand` and `--from-stand`, whose references are placed by heading only (a
robot-anchored goal needs the reference in the robot's world). The five-point deploy path is in this
repository's earlier release (commit aa40fe4). An export from before (keypoint goal
version 1-2) still runs: the node runs the graph cut down to the one
output it reads (`pruned`), so its five-point inputs and its `action_keypoint` output
drop out. An export with the human interface (`<robot>.json` `human_interface`) runs the
same way: its human inputs and `action_human` drop out.

`--interface human` acts through `action_human` instead: the reference is a human window, the 24 SMPL
slots on the uniform skeleton over the K future frames (`deploy/sim2sim/human.py`), and
the node builds the goal features, the `--human-points` slot mask (`full` or `five`, the
settings the current runs train), every frame visible, and the heading block (the human
Spine3 against the robot torso, from the IMU and the joint encoders) on. The sources are
the human streamer (`--source teleop`: `streamer.py --target human`, PICO on the uniform
skeleton, no GMR; `teleop.HumanTeleopReference`), the uniform skeleton's neutral stand
(`--source stand`) or a training clip's human sidecar (`--source clip --human-clip
<human50>/<date>/<stem>.npz`, frame k with the robot clip's frame k: the sim starts the
robot on the robot clip). Only exports of the node's keypoint goal version have it; a
direct human policy (`--policy-goal human`: `action_human` only, no `action_motion`)
runs only this way (`DIRECT_HUMAN_EXPORT_ERROR` otherwise).

  uv run --group export python -m deploy.sim2sim.policy_node <deploy_dir> --robot g1 \\
    --clip <assets>/clips/g1/<clip>.npz
  uv run --group export python -m deploy.sim2sim.policy_node <deploy_dir> --robot x2 \\
    --source teleop [--streamer tcp://HOST:5556]
  uv run --group export python -m deploy.sim2sim.policy_node <deploy_dir> --robot g1
  uv run --group export python -m deploy.sim2sim.policy_node <deploy_dir> --robot g1 \\
    --interface human --human-points five --source teleop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

from deploy.hardware.pose_source import NO_POSE_SOURCE
from deploy.sim2sim import protocol
from deploy.sim2sim.human import (
  heading_features,
  human_features,
  human_stand,
  join_frames,
  point_mask,
  turn_human,
)
from deploy.sim2sim.obs import (
  GOAL_K,
  PROPRIO_DIM,
  History,
  Robot,
  goal_motion,
  proprio_frame,
  reference_window,
)
from deploy.vr_teleop.constants import FPS, PORT, STAND_BLEND_S

# Keypoint goal version of the human interface (a copy of morphmimic.constants').
KP_GOAL_VERSION = 10
# onnxruntime's default pool takes every core and keeps it spinning after each call,
# which on the G1's Jetson (8 cores) slows the robot node and the SDK beside the policy;
# 4 threads run it as fast as 8 there.
ORT_THREADS = 4
# The reference frame the goal window starts at, relative to the control step. In the
# training env the robot starts in reference frame 0, but the motion command has
# already advanced to frame 1 when the first observation is made (mjlab updates the
# command once during the reset), and it advances once per step after that: at step t
# the policy has always seen frames t + 1 .. t + K.
REFERENCE_LEAD = 1


class ClipReference:
  """A recorded clip's windows: at step t, frames t + REFERENCE_LEAD onwards."""

  def __init__(self, states: np.ndarray) -> None:
    self.states = np.asarray(states, np.float64)

  def __len__(self) -> int:
    return len(self.states)

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    return reference_window(self.states, step_id + REFERENCE_LEAD)

  def reset(self) -> None:
    pass


def placed(states: np.ndarray, yaw: float, at: np.ndarray) -> np.ndarray:
  """Reference states [T, 240] turned and moved so that frame 0's pelvis faces `yaw`
  (rad) at `at` (x, y)."""
  from deploy.sim2sim.teleop import heading, turn_states

  out = turn_states(states, yaw - heading(states[0, 100:104]), states[0, 58:60])
  pos = out[:, 58:100].reshape(len(out), -1, 3)
  pos[..., :2] += at
  out[:, 58:100] = pos.reshape(len(out), -1)
  return out


class CuedClipReference:
  """A clip on cue, from the neutral stand: the robot's case on hardware, where it
  starts in the stand facing wherever the IMU's yaw says (the remote's A runs the
  policy, Y plays the clip).

  Until a cue, and again after the clip, the window is the neutral stand (`stand.json`)
  facing the robot. A cue (`cue`: the state message's count of them went up) while
  standing turns the clip so its frame-0 pelvis heading is the robot's (as teleop's POSE
  entry) and plays `lead_in_s` from the stand into frame 0, the clip, then `return_s`
  from its last frame back into the stand. The stand is placed where the clip starts
  and ends: the goals are relative to each window's frame 0, so a jump in position
  would read as a step. A cue during the motion is ignored."""

  def __init__(self, states: np.ndarray, robot: Robot, lead_in_s: float = 2.0,
               return_s: float = STAND_BLEND_S) -> None:
    import deploy.sim2sim.teleop  # noqa: F401  Loaded now, not in the first control step.
    from deploy.vr_teleop.gmr_retarget import STAND_FILE

    self.states = np.asarray(states, np.float64)
    self.robot = robot
    self.lead_in, self.ret = round(lead_in_s * FPS), round(return_s * FPS)
    self.stand = np.asarray(json.loads(STAND_FILE.read_text())[robot.body]["state"])
    self.reset()

  def __len__(self) -> int:
    """Frames one cue plays: the lead-in, the clip and the return."""
    return self.lead_in + len(self.states) + self.ret

  def reset(self) -> None:
    self._cues, self._pending = 0, False
    self._idle: np.ndarray | None = None  # The stand window, between motions.
    self._played: np.ndarray | None = None
    self._start = 0

  def cue(self, step_id: int, cues: int) -> None:
    """`cues`: cues since the policy started (the robot side's counter)."""
    if cues <= self._cues:
      return
    self._cues = cues
    if self._played is not None or self._pending:
      print("[policy_node] cue ignored: the motion is still playing", flush=True)
    else:
      self._pending = True

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    from deploy.sim2sim.teleop import heading

    if self._idle is None:
      yaw = heading(self.robot.slot0_quat(imu_quat, q))
      self._idle = np.repeat(placed(self.stand[None], yaw, np.zeros(2)), GOAL_K, 0)
    if self._pending:
      self._pending = False
      self._played, self._start = self._motion(heading(self.robot.slot0_quat(imu_quat, q))), step_id
      print(f"[policy_node] cue: playing the clip ({len(self)} frames)", flush=True)
    if self._played is not None:
      frame = step_id - self._start + REFERENCE_LEAD
      if frame < len(self._played):
        return reference_window(self._played, frame)
      self._idle = np.repeat(self._played[-1:], GOAL_K, 0)
      self._played = None
      print("[policy_node] clip done: standing", flush=True)
    return self._idle

  def _motion(self, yaw: float) -> np.ndarray:
    from deploy.sim2sim.teleop import blend_states, heading

    start = self._idle[0]
    clip = placed(self.states, yaw, start[58:60])
    end = clip[-1]
    stand = placed(self.stand[None], heading(end[100:104]), end[58:60])[0]
    lead = [blend_states(start, clip[0], (i + 1) / self.lead_in) for i in range(self.lead_in)]
    back = [blend_states(end, stand, (i + 1) / self.ret) for i in range(self.ret)]
    return np.concatenate([np.reshape(x, (-1, clip.shape[1])) for x in (lead, clip, back)])


class StandReference:
  """The neutral stand (`stand.json`, as in teleop's IDLE) in every window frame, turned
  to the robot's heading at the first step after each reset: the goals compare the
  reference's orientations with the robot's full one, and the IMU's yaw is arbitrary."""

  def __init__(self, robot: Robot) -> None:
    import deploy.sim2sim.teleop  # noqa: F401  Loaded now, not in the first control step.

    self.robot = robot
    self.stand = self._stand()
    self._window: np.ndarray | None = None

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    if self._window is None:
      from deploy.sim2sim.teleop import heading

      yaw = heading(self.robot.slot0_quat(imu_quat, q))
      self._window = np.repeat(self._turned(self.stand, yaw)[None], GOAL_K, 0)
    return self._window

  def reset(self) -> None:
    self._window = None

  def _stand(self) -> np.ndarray:
    from deploy.vr_teleop.gmr_retarget import STAND_FILE

    return np.asarray(json.loads(STAND_FILE.read_text())[self.robot.body]["state"])

  @staticmethod
  def _turned(stand: np.ndarray, yaw: float) -> np.ndarray:
    from deploy.sim2sim.teleop import turn_states

    return turn_states(stand, yaw, np.zeros(2))


class HumanStandReference(StandReference):
  """`StandReference` for the human interface: the uniform skeleton's neutral stand
  (`human_stand`), turned to the robot's heading."""

  def _stand(self) -> np.ndarray:
    return human_stand()

  @staticmethod
  def _turned(stand: np.ndarray, yaw: float) -> np.ndarray:
    return turn_human(stand, yaw, np.zeros(2))


def human_clip(path: Path) -> np.ndarray:
  """A training clip's human sidecar (`human_pos_w`, `human_quat_w`) as human frames
  [T, 168], for `ClipReference`."""
  d = np.load(path)
  return join_frames(d["human_pos_w"].astype(np.float64), d["human_quat_w"].astype(np.float64))


def pruned(model, output: str) -> bytes:
  """The ONNX `model` cut down to `output` and the inputs it reads. onnxruntime runs the
  whole graph whichever outputs are asked for (`action_human` alone is 2.6 ms instead of
  3.4 on the G1's Jetson), and wants a feed for every input, read or not."""
  import onnx.utils

  producer = {t: node for node in model.graph.node for t in node.output}
  read, todo = set(), [output]
  while todo:
    t = todo.pop()
    if t not in read:
      read.add(t)
      todo.extend(producer[t].input if t in producer else ())
  inputs = [i.name for i in model.graph.input if i.name in read]
  return onnx.utils.Extractor(model).extract_model(inputs, [output]).SerializeToString()


DIRECT_HUMAN_EXPORT_ERROR = (
  "this export's policy reads the human goal (--policy-goal human): it has no"
  " action_motion; run it with --interface human"
)


ROBOT_ANCHOR_SOURCE_ERROR = (
  "--goal-anchor robot needs a reference in the robot's world: --source teleop, --source"
  " stand and --from-stand place their references by heading only; use --source clip"
  " without --from-stand (the only one for now)"
)


def check_anchor_source(goal_anchor: str, source: str, from_stand: float | None) -> None:
  """Refuses (RuntimeError) a robot-anchored export with a reference that is placed by
  heading only: teleop, the stand and a clip led in from the stand. Robot-anchored
  goals compare the reference with the robot's world position, so the reference must
  live in the robot's world (a clip played as recorded)."""
  if goal_anchor == "robot" and (source != "clip" or from_stand is not None):
    raise RuntimeError(ROBOT_ANCHOR_SOURCE_ERROR)


class PolicyRunner:
  """One robot's policy: measurements and a reference window in, joint targets out.
  `reference` gives the window (`ClipReference`, `teleop.TeleopReference`; human windows
  under `interface="human"`). The motion goal's anchor is the export's (`goal_anchor`);
  the human goal is the anchor's own heading frame plus the robot torso in the heading
  block, whatever the export's anchor. `human_points` is the human slot setting."""

  def __init__(self, deploy_dir: Path, robot: str, reference, interface: str = "motion",
               human_points: str = "full") -> None:
    import onnx
    import onnxruntime as ort

    doc = json.loads((Path(deploy_dir) / f"{robot}.json").read_text())
    self.robot = Robot.from_dict(doc["robot"])
    model = onnx.load(Path(deploy_dir) / "policy.onnx")
    self.interface = interface
    self.human_interface = bool(doc.get("human_interface", False))
    version = int(doc.get("kp_goal_version", 1))
    if interface == "human":
      if not self.human_interface or version != KP_GOAL_VERSION:
        raise RuntimeError(f"{deploy_dir}: {human_interface_error(self.human_interface, version)}")
      self.point_mask = point_mask(human_points)
    elif interface != "motion":
      raise ValueError(f"unknown interface {interface!r}")
    elif "action_motion" not in {o.name for o in model.graph.output}:
      raise RuntimeError(f"{deploy_dir}: {DIRECT_HUMAN_EXPORT_ERROR}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = ORT_THREADS
    self.session = ort.InferenceSession(pruned(model, f"action_{interface}"), options,
                                        providers=["CPUExecutionProvider"])
    read = {i.name for i in self.session.get_inputs()}
    self.inputs = {k: np.asarray(v, np.float32) for k, v in doc["inputs"].items() if k in read}
    self.goal_anchor = doc.get("goal_anchor", "reference")
    if self.goal_anchor not in ("reference", "robot"):
      raise ValueError(f"{robot}.json: unknown goal_anchor {self.goal_anchor!r}")
    if self.goal_anchor == "robot" and self.robot.fk_pos is None:
      raise ValueError(f"{robot}.json: the robot constants have no body offsets (`fk`"
                       " `pos`), which the robot goal anchor needs; re-export")
    if interface == "motion" and self.goal_anchor == "robot" and not isinstance(
        reference, ClipReference):
      raise RuntimeError(ROBOT_ANCHOR_SOURCE_ERROR)
    width = {i.name: i.shape[-1] for i in self.session.get_inputs()}["obs_history"]
    if width % PROPRIO_DIM:
      raise ValueError(f"obs_history width {width} is not a multiple of {PROPRIO_DIM}")
    self.history = History(width // PROPRIO_DIM)
    self.reference = reference
    self.last_action = np.zeros(29)
    self.prev_step = -1
    self.ahead: tuple | None = None  # `prepare`'s (step_id, window, human features)
    self._warm_up()

  def _warm_up(self) -> None:
    """One inference on zeros now, so the first control step does not pay for
    onnxruntime's first run."""
    types = {"tensor(float)": np.float32, "tensor(bool)": bool}
    feeds = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], types[i.type])
             for i in self.session.get_inputs()}
    feeds.update(self.inputs)
    self.session.run(None, feeds)

  def reset(self) -> None:
    self.history.reset()
    self.last_action = np.zeros(29)
    self.ahead = None

  def prepare(self, step_id: int, quat: np.ndarray, q: np.ndarray) -> None:
    """The reference window of control step `step_id` (and its human features), built
    before that step's state arrives so `step` skips them: off the state-to-action path.
    The window reads the state only for the robot heading when a mode is entered; the
    last measurements stand in for it. A reference that takes cues (they come with the
    state) is left to `step`."""
    if hasattr(self.reference, "cue"):
      return
    ref = self.reference.window(step_id, quat / np.linalg.norm(quat), q)
    feats = (human_features(ref, self.point_mask)[None].astype(np.float32)
             if self.interface == "human" else None)
    self.ahead = (step_id, ref, feats)

  def step(self, step_id: int, quat: np.ndarray, gyro: np.ndarray, q: np.ndarray,
           dq: np.ndarray, cues: int = 0,
           pose: tuple[np.ndarray, float] | None = None) -> np.ndarray:
    """Native joint targets [J] for the measurements of control step `step_id` (steps
    since the last reset, when the robot stood in reference frame 0); `cues` (the
    remote's Y presses since then) go to a reference that takes them. `pose` is the
    floating base's (world position [3], yaw), read under the robot goal anchor only,
    where a missing or NaN pose raises RuntimeError (`NO_POSE_SOURCE`)."""
    no_pose = pose is None or not (np.isfinite(pose[0]).all() and np.isfinite(pose[1]))
    if self.interface == "motion" and self.goal_anchor == "robot" and no_pose:
      raise RuntimeError(NO_POSE_SOURCE)
    if step_id < self.prev_step:
      self.reset()
      self.reference.reset()
    self.prev_step = step_id
    if hasattr(self.reference, "cue"):
      self.reference.cue(step_id, cues)
    quat = quat / np.linalg.norm(quat)  # the IMU's
    frame = proprio_frame(self.robot, q, dq, quat, gyro, self.last_action)
    obs = self.history.push(frame)
    if self.ahead is not None and self.ahead[0] == step_id:
      _, ref, feats = self.ahead
    else:
      ref, feats = self.reference.window(step_id, quat, q), None
    self.ahead = None
    self.last_ref = ref
    feeds = {"obs_history": obs[None].astype(np.float32), **self.inputs,
             **(self._human_goal(ref, quat, q, feats) if self.interface == "human"
                else self._motion_goal(ref, quat, q, pose))}
    action = self.session.run([f"action_{self.interface}"], feeds)[0][0].astype(np.float64)
    self.last_action = action
    return self.robot.targets(action)

  def _motion_goal(self, ref: np.ndarray, quat: np.ndarray, q: np.ndarray,
                   pose: tuple[np.ndarray, float] | None) -> dict[str, np.ndarray]:
    if self.goal_anchor == "robot":
      anchor_pos, anchor_quat = self.robot.anchor_pose(pose[0], pose[1], quat, q)
    else:
      anchor_pos, anchor_quat = None, self.robot.anchor_quat(quat, q)
    return {"goal_motion": goal_motion(ref, anchor_quat, anchor_pos, self.goal_anchor)[None]}

  def _human_goal(self, ref: np.ndarray, quat: np.ndarray, q: np.ndarray,
                  feats: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The human inputs of a human window `ref` [K, 168]: the features, the slot mask,
    every frame visible (the jitter buffer makes the preview real), the heading block
    against the robot torso, on."""
    torso = self.robot.anchor_quat(quat, q)
    return {
      "goal_human": (feats if feats is not None
                     else human_features(ref, self.point_mask)[None].astype(np.float32)),
      "human_point_mask": self.point_mask[None],
      "human_frame_mask": np.ones((1, GOAL_K), bool),
      "human_heading": heading_features(torso, ref, self.point_mask)[None].astype(np.float32),
      "human_heading_mask": np.ones(1, bool),
    }


def human_interface_error(has_human: bool, version: int) -> str:
  """Why an export cannot run `--interface human`."""
  if not has_human:
    return "this export has no human interface (no action_human): its checkpoint has no human encoder"
  return (f"this export's human interface is keypoint goal version {version}; this node"
          f" builds version {KP_GOAL_VERSION}'s goal")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("deploy_dir", type=Path,
                      help="export_onnx.py output: policy.onnx and <robot>.json")
  parser.add_argument("--robot", required=True)
  parser.add_argument("--source", choices=["clip", "teleop", "stand"], default=None,
                      help="default: clip with --clip, else stand")
  parser.add_argument("--clip", type=Path, help="reference clip .npz (--source clip)")
  parser.add_argument("--from-stand", type=float, default=None, metavar="S",
                      help="clip: the robot stands until a cue (hardware: Y), then the"
                      " clip, turned to its heading, led in over S s")
  parser.add_argument("--streamer", default=f"tcp://127.0.0.1:{PORT}",
                      help="the streamer's endpoint (--source teleop)")
  parser.add_argument("--interface", choices=["motion", "human"], default="motion",
                      help="act through the robot reference (action_motion, GMR in teleop)"
                      " or the human skeleton (action_human, no GMR)")
  parser.add_argument("--human-points", choices=["full", "five", "six", "three"],
                      default="full", help="--interface human: the slot setting (the"
                      " current runs train full and five)")
  parser.add_argument("--human-clip", type=Path, default=None,
                      help="--interface human, --source clip: the clip's human sidecar"
                      " (<human50>/<date>/<stem>.npz)")
  parser.add_argument("--bind-host", default=protocol.HOST,
                      help="address the state port listens on (0.0.0.0 for a remote robot)")
  parser.add_argument("--robot-host", default=protocol.HOST,
                      help="where the robot side runs (sim_node, or deploy.hardware.g1_node)")
  parser.add_argument("--idle-exit-s", type=float, default=0.0,
                      help="exit after this long without a state message (0: never)")
  args = parser.parse_args()

  human = args.interface == "human"
  if args.source is None:
    args.source = "clip" if (args.human_clip if human else args.clip) is not None else "stand"
  if human and args.from_stand is not None:
    parser.error("--from-stand leads into a robot reference clip: not with --interface human")
  doc = json.loads((args.deploy_dir / f"{args.robot}.json").read_text())
  if not human:
    check_anchor_source(doc.get("goal_anchor", "reference"), args.source, args.from_stand)
  robot = Robot.from_dict(doc["robot"])
  # The sim draws robot reference windows only, so a human window is not sent.
  display = None
  if args.source == "clip":
    clip = args.human_clip if human else args.clip
    if clip is None:
      parser.error(f"--source clip needs {'--human-clip' if human else '--clip'}")
    if human:
      reference = ClipReference(human_clip(clip))
    elif args.from_stand is None:
      reference = ClipReference(np.load(clip)["states"])
    else:
      reference = CuedClipReference(np.load(clip)["states"], robot, args.from_stand)
      display = protocol.sender(protocol.DISPLAY_PORT, args.robot_host)
    what = f"{len(reference)} {'human' if human else 'reference'} frames"
  elif args.source == "stand":
    reference = HumanStandReference(robot) if human else StandReference(robot)
    what = "the neutral stand"
  else:
    from deploy.sim2sim.teleop import HumanTeleopReference, TeleopReference

    reference = (HumanTeleopReference if human else TeleopReference)(robot, args.streamer)
    what = f"live {'human' if human else 'reference'} frames from {args.streamer}"
  if not human and args.source != "clip":
    display = protocol.sender(protocol.DISPLAY_PORT, args.robot_host)
  runner = PolicyRunner(args.deploy_dir, args.robot, reference, args.interface,
                        args.human_points)
  j = runner.robot.num_native
  state_in = protocol.receiver(protocol.STATE_PORT, args.bind_host, timeout=0.5)
  action_out = protocol.sender(protocol.ACTION_PORT, args.robot_host, timeout=0.1)
  goal = (f"human goal, {args.human_points} points" if human
          else f"goal anchor {runner.goal_anchor}")
  print(f"[policy_node] {args.robot}: {what}; {goal}; state <- {protocol.STATE_PORT}, "
        f"action -> {protocol.ACTION_PORT}", flush=True)
  last, n, t_log, t_cost = time.monotonic(), 0, time.monotonic(), 0.0
  try:
    while True:
      try:
        data = protocol.recv_latest(state_in)
      except zmq.Again:
        if args.idle_exit_s and time.monotonic() - last > args.idle_exit_s:
          break
        continue
      last = time.monotonic()
      step_id, quat, gyro, q, dq = protocol.unpack_state(data, j)
      targets = runner.step(step_id, quat, gyro, q, dq, protocol.state_cues(data, j),
                            protocol.state_pose(data, j))
      try:
        action_out.send(protocol.pack_action(step_id, targets))
      except zmq.Again:  # The robot side is gone; its next state message resumes us.
        pass
      if display is not None:  # Drawing only: never wait for it (no sim on hardware).
        try:
          display.send(protocol.pack_window(runner.last_ref), zmq.NOBLOCK)
        except zmq.Again:
          pass
      t_cost += time.monotonic() - last
      n += 1
      runner.prepare(step_id + 1, quat, q)  # After the action went out: off its path.
      if time.monotonic() - t_log > 5.0:
        print(f"[policy_node] step {step_id}, {1e3 * t_cost / n:.2f} ms per step", flush=True)
        t_log, n, t_cost = time.monotonic(), 0, 0.0
  except KeyboardInterrupt:
    pass
  finally:
    state_in.close()
    action_out.close()
    if display is not None:
      display.close()
    if hasattr(reference, "close"):
      reference.close()


if __name__ == "__main__":
  main()
