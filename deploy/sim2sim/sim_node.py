"""The robot, simulated in plain CPU MuJoCo (`mujoco` and numpy; no mjlab, no torch).

Loads a robot's exported scene XML (`export_assets.py`: the trained model, readable),
starts it in the clip's first reference pose and plays the clip in lock step with
`policy_node.py` over the `protocol` messages, as the robot would: each control step it
sends the joint encoders and the IMU (the IMU body's orientation and gyro), and the
floating base's true pose (`SimPoseSource`, the pose a state estimator gives on
hardware, read only by a robot-anchor policy), waits for the policy's targets for that
step, applies them (optionally delayed) and steps the physics `decimation` times. The
targets go to the position actuators, or, for the DC-motor joints (the X2's, as in
training), into an explicit PD clipped to the motor's torque-speed curve, recomputed
each physics step (`SimRobot.dc_torque`).

It also measures how the robot tracks the clip, from simulator truth the policy never
sees: failure as in training (anchor height off by more than 0.25 m, or tilt off by more
than 0.8 in the gravity z component), the joint and tracked-body errors, and the drift
of the anchor in the reference heading frame. `--viewer native` shows the run live and
`--video` renders it, both with what the policy tracks drawn in: the reference as a
translucent ghost.

Teleop (`--teleop`, with `policy_node.py --source teleop`): no clip and no reference to
score against; the robot starts in the neutral stand and runs in real time until the
viewer closes (or `--max-steps`), a fall resetting it to the stand. The overlay is the
window the policy node reports tracking (`LiveOverlay`), as a trace of its pelvis,
ankles and wrists.

  uv run --group export python -m deploy.sim2sim.sim_node <assets>/g1.xml --robot g1 \\
    --clip <assets>/clips/g1/<clip>.npz [--video out.mp4] [--delay-ms 20]
  uv run --group export python -m deploy.sim2sim.sim_node <assets>/x2.xml --robot x2 \\
    --teleop [--viewer native]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq

from deploy.hardware.pose_source import SimPoseSource
from deploy.sim2sim import protocol
from deploy.sim2sim.obs import ANCHOR_IDX, Robot, quat_apply_inverse, quat_mul, yaw_quat
from deploy.vr_teleop.constants import FALL_HEIGHT, FALL_TILT_DEG

_PREFIX = "robot/"
Z_FAIL = 0.25
TILT_FAIL = 0.8


def body_gyro(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
  """Angular velocity of `body` in its own frame, as an IMU on it reads it. XBODY is the
  body frame; BODY would be the inertial frame, which on the X2 pelvis is rotated away
  from the body frame."""
  vel = np.zeros(6)
  mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, vel, 1)
  return vel[:3].copy()


class SimRobot:
  """The simulated robot: model, state, actuators and the measurements it reports."""

  def __init__(self, xml: Path, robot: Robot) -> None:
    self.robot = robot
    self.model = mujoco.MjModel.from_xml_path(str(xml))
    self.data = mujoco.MjData(self.model)
    m = self.model

    def jid(name: str) -> int:
      i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, _PREFIX + name)
      if i < 0:
        raise ValueError(f"joint {name} not in {xml}")
      return i

    joints = [jid(n) for n in robot.native_joint_names]
    self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
    self.dadr = np.array([m.jnt_dofadr[j] for j in joints])
    native_of_joint = {j: i for i, j in enumerate(joints)}
    self.act_native = np.array([native_of_joint[m.actuator_trnid[a, 0]] for a in range(m.nu)])
    self.bodies = np.array([
      mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, _PREFIX + b) for b in robot.tracked_bodies
    ])
    if (self.bodies < 0).any():
      raise ValueError(f"tracked bodies missing from {xml}")
    self.imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, _PREFIX + robot.imu_body)
    if not math.isclose(m.opt.timestep, robot.physics_dt):
      raise ValueError(f"{xml}: timestep {m.opt.timestep}, robot constants {robot.physics_dt}")
    # The DC-motor joints (mjlab `DcMotorActuatorCfg`) are torque actuators: their torque
    # is the explicit PD, clipped to the torque-speed curve, computed each physics step.
    # Their constants are the simulated robot's: `robot`'s, or, from a deployment folder
    # exported before the robot had DC motors, the baked `robots/<name>.json` the
    # export wrote with this XML.
    self.motor = m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE
    joint = self.act_native[self.motor]
    if self.motor.any():
      act = robot if robot.dc_saturation is not None else Robot.load(robot.name)
      if (act.native_joint_names != robot.native_joint_names or act.dc_saturation is None
          or not (act.dc_velocity_limit[joint] > 0).all()):
        raise ValueError(f"{xml} has torque actuators but robots/{robot.name}.json no DC motors")
      self._dc = (self.qadr[joint], self.dadr[joint], act.kp[joint], act.kd[joint],
                  act.effort_limit[joint], act.dc_saturation[joint],
                  act.dc_velocity_limit[joint])
      self._dc_law = act.dc_law
    self.targets = np.zeros(m.nu)
    self.pose_source = SimPoseSource(m, self.data)

  def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] = qpos
    self.data.qvel[:] = qvel
    mujoco.mj_forward(self.model, self.data)
    self.set_targets(self.data.qpos[self.qadr])

  def set_targets(self, targets: np.ndarray) -> None:
    self.targets = targets[self.act_native]
    self.data.ctrl[~self.motor] = self.targets[~self.motor]

  def dc_torque(self) -> np.ndarray:
    """The DC-motor joints' torque now: kp (target - q) - kd qd, within [-limit, limit]
    and the torque-speed curve, mjlab's `DcMotorActuator` sat (+-1 - qd / v_max) or
    Booster's two-sided +-max(0, sat (1 - |qd| / v_max)) (`robots.booster_motor`)."""
    qadr, dadr, kp, kd, limit, sat, vmax = self._dc
    q, qd = self.data.qpos[qadr], self.data.qvel[dadr]
    torque = kp * (self.targets[self.motor] - q) - kd * qd
    if self._dc_law == "booster":
      bound = np.minimum(np.maximum(sat * (1 - np.abs(qd) / vmax), 0.0), limit)
      return np.clip(torque, -bound, bound)
    v = np.clip(qd, -vmax * (1 + limit / sat), vmax * (1 + limit / sat))
    upper = np.minimum(sat * (1 - v / vmax), limit)
    lower = np.maximum(sat * (-1 - v / vmax), -limit)
    return np.clip(torque, lower, upper)

  def measure(self) -> tuple[np.ndarray, ...]:
    """(IMU quat, IMU gyro in its own frame, q [J], dq [J]): what the robot's encoders
    and IMU report."""
    return (self.data.xquat[self.imu].copy(), body_gyro(self.model, self.data, self.imu),
            self.data.qpos[self.qadr].copy(), self.data.qvel[self.dadr].copy())

  def step(self) -> None:
    if self.motor.any():
      self.data.ctrl[self.motor] = self.dc_torque()
    mujoco.mj_step(self.model, self.data)

  def bodies_pose(self) -> tuple[np.ndarray, np.ndarray]:
    return self.data.xpos[self.bodies].copy(), self.data.xquat[self.bodies].copy()


def _split(states: np.ndarray):
  q = states[..., :29]
  pos = states[..., 58:100].reshape(*states.shape[:-1], 14, 3)
  quat = states[..., 100:156].reshape(*states.shape[:-1], 14, 4)
  return q, pos, quat


class Tracker:
  """Tracking errors of the robot against the reference, per control step."""

  def __init__(self, sim: SimRobot, states: np.ndarray) -> None:
    self.sim, self.states = sim, states
    self.rows: list[dict] = []
    self.failed_at: int | None = None

  def record(self, frame: int) -> bool:
    """Errors at reference frame `frame`; False once the robot has failed."""
    ref_q, ref_pos, ref_quat = _split(self.states[frame])
    pos, quat = self.sim.bodies_pose()
    q = self.sim.robot.to_canonical(self.sim.data.qpos[self.sim.qadr])
    down = np.array([0.0, 0.0, -1.0])
    e_z = abs(ref_pos[ANCHOR_IDX, 2] - pos[ANCHOR_IDX, 2])
    e_tilt = abs(quat_apply_inverse(ref_quat[ANCHOR_IDX], down)[2]
                 - quat_apply_inverse(quat[ANCHOR_IDX], down)[2])
    ref_yaw = yaw_quat(ref_quat[0])
    drift = quat_apply_inverse(ref_yaw, np.r_[pos[ANCHOR_IDX, :2] - ref_pos[ANCHOR_IDX, :2], 0.0])
    # Body positions relative to the anchor, both in their own heading frames.
    rel = quat_apply_inverse(yaw_quat(quat[ANCHOR_IDX])[None], pos - pos[ANCHOR_IDX])
    ref_rel = quat_apply_inverse(ref_yaw[None], ref_pos - ref_pos[ANCHOR_IDX])
    heading = math.atan2(2 * (quat[0, 0] * quat[0, 3] + quat[0, 1] * quat[0, 2]),
                         1 - 2 * (quat[0, 2] ** 2 + quat[0, 3] ** 2))
    ref_heading = math.atan2(2 * (ref_quat[0, 0] * ref_quat[0, 3] + ref_quat[0, 1] * ref_quat[0, 2]),
                             1 - 2 * (ref_quat[0, 2] ** 2 + ref_quat[0, 3] ** 2))
    self.rows.append({
      "frame": frame,
      "joint_err": float(np.sqrt(np.mean((q - ref_q) ** 2))),
      "body_err": float(np.mean(np.linalg.norm(rel - ref_rel, axis=-1))),
      "drift_along": float(drift[0]),
      "drift_across": float(drift[1]),
      "heading_err": float(math.remainder(heading - ref_heading, 2 * math.pi)),
      "e_z": float(e_z),
      "e_tilt": float(e_tilt),
    })
    if self.failed_at is None and (e_z > Z_FAIL or e_tilt > TILT_FAIL):
      self.failed_at = frame
    return self.failed_at is None

  def summary(self, clip: str, robot: str, length: int) -> dict:
    rows = self.rows
    last = rows[-1]
    return {
      "robot": robot,
      "clip": clip,
      "frames": length,
      "result": "completed" if self.failed_at is None else f"failed at frame {self.failed_at}",
      "joint_err_rms_rad": float(np.mean([r["joint_err"] for r in rows])),
      "body_err_mean_m": float(np.mean([r["body_err"] for r in rows])),
      "end_drift_m": float(math.hypot(last["drift_along"], last["drift_across"])),
      "end_drift_along_m": last["drift_along"],
      "end_drift_across_m": last["drift_across"],
      "end_heading_err_deg": math.degrees(last["heading_err"]),
    }


# The teleop window's trace (`LiveOverlay`): per body (pelvis, left ankle, right ankle,
# left wrist, right wrist) a sphere at the current target, a line through the preview
# frames, a fading trail.
TRACE_IDX = (0, 3, 6, 10, 13)
TRACE_COLORS = ((1.0, 0.85, 0.1), (1.0, 0.45, 0.1), (0.1, 0.8, 0.9), (0.9, 0.15, 0.25),
                (0.2, 0.4, 1.0))
TARGET_RADIUS, PREVIEW_RADIUS, TRAIL_RADIUS = 0.04, 0.016, 0.010
TRAIL_ALPHA = (0.3, 0.9)
GHOST_RGBA = (0.3, 0.6, 1.0, 0.3)
# Teleop (`run_teleop`): the stand's feet start this far above the floor.
STAND_CLEARANCE = 0.01


def _add_geom(scene: mujoco.MjvScene) -> mujoco.MjvGeom | None:
  if scene.ngeom >= scene.maxgeom:
    return None
  scene.ngeom += 1
  return scene.geoms[scene.ngeom - 1]


def _sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba) -> None:
  g = _add_geom(scene)
  if g is not None:
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        pos, np.eye(3).ravel(), np.asarray(rgba, np.float32))


def _segment(scene: mujoco.MjvScene, a: np.ndarray, b: np.ndarray, radius: float, rgba) -> None:
  g = _add_geom(scene)
  if g is not None:
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)


class Overlay:
  """What the policy is asked to track, drawn into a scene: the reference as a
  translucent ghost."""

  def __init__(self, sim: SimRobot, ref, mode: str = "ghost") -> None:
    if mode != "ghost":
      raise ValueError(f"overlay must be ghost, got {mode!r}")
    self.sim = sim
    self.ghost_qpos = ref["ghost_qpos"]
    self.ghost = mujoco.MjData(sim.model)
    self.option, self.perturb = mujoco.MjvOption(), mujoco.MjvPerturb()

  def draw(self, scene: mujoco.MjvScene, step: int) -> None:
    self.ghost.qpos[:] = self.ghost_qpos[min(step, len(self.ghost_qpos) - 1)]
    mujoco.mj_kinematics(self.sim.model, self.ghost)
    start = scene.ngeom
    mujoco.mjv_addGeoms(self.sim.model, self.ghost, self.option, self.perturb,
                        mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    for g in range(start, scene.ngeom):
      scene.geoms[g].rgba[:] = GHOST_RGBA


def draw_trace(scene: mujoco.MjvScene, future: np.ndarray, trail: list[np.ndarray]) -> None:
  """The trace of the `TRACE_IDX` bodies: `future` [K, 5, 3] (the targets and their
  preview) and the fading `trail` of past targets."""
  trail = trail or [future[0]]
  alphas = np.linspace(*TRAIL_ALPHA, max(1, len(trail) - 1))
  for p, rgb in enumerate(TRACE_COLORS):
    for i in range(len(trail) - 1):
      _segment(scene, trail[i][p], trail[i + 1][p], TRAIL_RADIUS, (*rgb, alphas[i]))
    prev = future[0][p]
    for k in range(1, len(future)):
      _segment(scene, prev, future[k][p], PREVIEW_RADIUS, (*rgb, 0.95))
      prev = future[k][p]
    _sphere(scene, future[0][p], TARGET_RADIUS, (*rgb, 1.0))


class LiveOverlay:
  """The window the policy node tracks in teleop (`protocol.DISPLAY_PORT`), drawn as the
  trace of its pelvis, ankles and wrists (`draw_trace`). The goals are position-free, so
  the window is drawn where they put it: its frame-0 pelvis at the robot's pelvis (x, y)
  of this step."""

  def __init__(self, sim: SimRobot, trail_len: int = 50) -> None:
    self.sim, self.trail_len = sim, trail_len
    self.future: np.ndarray | None = None
    self.trail: list[np.ndarray] = []

  def update(self, window: np.ndarray) -> None:
    pos = _split(window)[1][:, list(TRACE_IDX)]
    robot_xy = self.sim.data.xpos[self.sim.bodies[0], :2]
    pos = pos + np.array([*(robot_xy - pos[0, 0, :2]), 0.0])
    self.future = pos
    self.trail = (self.trail + [pos[0]])[-self.trail_len:]

  def draw(self, scene: mujoco.MjvScene, step: int) -> None:
    if self.future is not None:
      draw_trace(scene, self.future, self.trail)


class Video:
  """Offscreen frames following the anchor, with the overlay."""

  def __init__(self, sim: SimRobot, overlay: Overlay, width: int = 1280,
               height: int = 720) -> None:
    self.sim, self.overlay = sim, overlay
    self.renderer = mujoco.Renderer(sim.model, height, width)
    self.camera = mujoco.MjvCamera()
    self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.0, 135.0, -20.0
    self.frames: list[np.ndarray] = []

  def capture(self, step: int) -> None:
    sim = self.sim
    self.camera.lookat[:] = sim.data.xpos[sim.bodies[ANCHOR_IDX]]
    self.renderer.update_scene(sim.data, self.camera)
    self.overlay.draw(self.renderer.scene, step)
    self.frames.append(self.renderer.render().copy())

  def save(self, path: Path, fps: float) -> None:
    import mediapy

    mediapy.write_video(str(path), self.frames, fps=fps)


class LiveView:
  """MuJoCo's interactive viewer, following the anchor, with the overlay. Needs a
  display."""

  def __init__(self, sim: SimRobot, overlay: Overlay) -> None:
    import mujoco.viewer

    self.sim, self.overlay = sim, overlay
    self.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    self.viewer.cam.trackbodyid = int(sim.bodies[ANCHOR_IDX])
    self.viewer.cam.distance, self.viewer.cam.azimuth = 3.0, 135.0
    self.viewer.cam.elevation = -20.0

  def show(self, step: int) -> None:
    with self.viewer.lock():
      self.viewer.user_scn.ngeom = 0
      self.overlay.draw(self.viewer.user_scn, step)
    self.viewer.sync()

  def wait_closed(self) -> None:
    print("[sim_node] clip done; close the viewer window to exit", flush=True)
    while self.viewer.is_running():
      time.sleep(0.1)


def run(xml: Path, robot: Robot, clip: Path, delay_ms: float = 0.0,
        video: Path | None = None, timeout_s: float = 5.0, realtime: bool = False,
        viewer: bool = False, overlay: str = "ghost") -> dict:
  """Plays `clip` with the policy node answering on the protocol ports; the summary.
  `viewer` shows the run live (paced at real time); `overlay` is what the video and the
  viewer draw: the reference `ghost`."""
  ref = np.load(clip)
  states = ref["states"].astype(np.float64)
  sim = SimRobot(xml, robot)
  sim.reset(ref["init_qpos"], ref["init_qvel"])
  tracker = Tracker(sim, states)
  drawn = Overlay(sim, ref, overlay)
  recorder = Video(sim, drawn) if video else None
  live = LiveView(sim, drawn) if viewer else None
  realtime = realtime or viewer
  lag = round(delay_ms / 1e3 / robot.physics_dt)
  pending: list[np.ndarray] = []
  hold = sim.data.qpos[sim.qadr].copy()
  action_in = protocol.receiver(protocol.ACTION_PORT, timeout=timeout_s)
  state_out = protocol.sender(protocol.STATE_PORT)

  def show(step: int) -> None:
    if recorder:
      recorder.capture(step)
    if live:
      live.show(step)

  try:
    tracker.record(0)
    show(0)
    for step in range(len(states) - 1):
      t0 = time.monotonic()
      quat, gyro, q, dq = sim.measure()
      pos, yaw = sim.pose_source.pose()
      state_out.send(protocol.pack_state(step, quat, gyro, q, dq, pelvis_pos=pos, pelvis_yaw=yaw))
      while True:
        try:
          data = action_in.recv()
        except zmq.Again:
          raise RuntimeError(f"no action for step {step} within {timeout_s} s:"
                             " is policy_node.py running?") from None
        got, targets = protocol.unpack_action(data, robot.num_native)
        if got == step:
          break
      for _ in range(robot.decimation):
        pending.append(targets)
        if len(pending) > lag:
          hold = pending.pop(0)
        sim.set_targets(hold)
        sim.step()
      mujoco.mj_forward(sim.model, sim.data)
      alive = tracker.record(step + 1)
      show(step + 1)
      if not alive:
        break
      if live and not live.viewer.is_running():
        break
      if realtime:
        time.sleep(max(0.0, robot.control_dt - (time.monotonic() - t0)))
  finally:
    action_in.close()
    state_out.close()
  summary = tracker.summary(str(ref["clip"]), robot.name, len(states))
  summary["delay_ms"] = delay_ms
  if recorder:
    recorder.save(video, 1.0 / robot.control_dt)
    summary["video"] = str(video)
  if live:
    live.wait_closed()
  return summary


def stand_qpos(model: mujoco.MjModel, sim: SimRobot, stand: dict,
               yaw_deg: float = 0.0) -> np.ndarray:
  """The model's qpos with the robot in the neutral stand (`stand.json`), its feet just
  above the floor, turned by `yaw_deg` about the vertical."""
  qpos = model.qpos0.copy()
  free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
  adr = model.jnt_qposadr[free]
  qpos[adr:adr + 3] = np.asarray(stand["base_pos"]) + [0.0, 0.0, STAND_CLEARANCE]
  turn = np.array([math.cos(math.radians(yaw_deg) / 2), 0.0, 0.0,
                   math.sin(math.radians(yaw_deg) / 2)])
  qpos[adr + 3:adr + 7] = quat_mul(turn, np.asarray(stand["base_quat"], float))
  qpos[sim.qadr] = [stand["joint_pos"][n] for n in sim.robot.native_joint_names]
  return qpos


def _planar_pose(sim: SimRobot) -> tuple[float, float, float]:
  """The pelvis-equivalent's x, y and heading (rad)."""
  w, x, y, z = sim.data.xquat[sim.bodies[0]]
  pos = sim.data.xpos[sim.bodies[0]]
  return float(pos[0]), float(pos[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def fallen(sim: SimRobot) -> bool:
  """The pelvis-equivalent below `FALL_HEIGHT` or tilted over `FALL_TILT_DEG` (the
  teleop sims' fall check)."""
  quat = sim.data.xquat[sim.bodies[0]]
  up_z = quat_apply_inverse(quat, np.array([0.0, 0.0, 1.0]))[2]
  return (sim.data.xpos[sim.bodies[0], 2] < FALL_HEIGHT
          or up_z < math.cos(math.radians(FALL_TILT_DEG)))


def run_teleop(xml: Path, robot: Robot, delay_ms: float = 0.0, video: Path | None = None,
               timeout_s: float = 5.0, viewer: bool = False,
               max_steps: int | None = None, start_yaw_deg: float = 0.0,
               cue_steps: tuple[int, ...] = (), trace: Path | None = None) -> dict:
  """The robot driven live by the policy node (`--source teleop`), from the neutral
  stand, in real time, until the viewer closes or after `max_steps` control steps. A
  fall resets the robot to the stand and the step count to 0, which the policy node
  takes as a reset (it holds OFF until the operator recalibrates). `cue_steps`: the
  steps (since the start or a fall) at which the remote's Y is pressed. `trace`: write
  an npz with, per control step, the robot's canonical joint angles `q`, its IMU
  quaternion `quat` (wxyz), `ref`, frame 0 of the reference window the policy node
  tracked (240-d, NaN before the first window; always NaN under the human interface,
  whose window the sim is not sent), and the simulator's full `qpos` (for rendering)."""
  from deploy.vr_teleop.gmr_retarget import STAND_FILE

  stand = json.loads(STAND_FILE.read_text())[robot.body]
  sim = SimRobot(xml, robot)
  init = stand_qpos(sim.model, sim, stand, start_yaw_deg)
  sim.reset(init, np.zeros(sim.model.nv))
  drawn = LiveOverlay(sim)
  recorder = Video(sim, drawn) if video else None
  live = LiveView(sim, drawn) if viewer else None
  lag = round(delay_ms / 1e3 / robot.physics_dt)
  pending: list[np.ndarray] = []
  hold = sim.data.qpos[sim.qadr].copy()
  action_in = protocol.receiver(protocol.ACTION_PORT, timeout=timeout_s)
  display_in = protocol.receiver(protocol.DISPLAY_PORT, timeout=0.0)
  state_out = protocol.sender(protocol.STATE_PORT)
  step, total, falls, t_start = 0, 0, 0, time.monotonic()
  start = _planar_pose(sim)
  joint_err: list[float] = []  # Against frame 0 of the window the policy node tracked.
  rows: dict[str, list] = {"step": [], "q": [], "quat": [], "ref": [], "qpos": []}
  last_ref = np.full(240, np.nan)
  print(f"[sim_node] {robot.name}: teleop, standing; {'close the viewer' if viewer else 'Ctrl-C'}"
        " to stop", flush=True)
  try:
    while max_steps is None or total < max_steps:
      t0 = time.monotonic()
      quat, gyro, q, dq = sim.measure()
      cues = sum(step >= c for c in cue_steps)
      pos, yaw = sim.pose_source.pose()
      state_out.send(protocol.pack_state(step, quat, gyro, q, dq, cues, pos, yaw))
      while True:
        try:
          got, targets = protocol.unpack_action(action_in.recv(), robot.num_native)
        except zmq.Again:
          raise RuntimeError(f"no action for step {step} within {timeout_s} s:"
                             " is policy_node.py running?") from None
        if got == step:
          break
      for _ in range(robot.decimation):
        pending.append(targets)
        if len(pending) > lag:
          hold = pending.pop(0)
        sim.set_targets(hold)
        sim.step()
      mujoco.mj_forward(sim.model, sim.data)
      try:
        window = protocol.unpack_window(protocol.recv_latest(display_in))
        drawn.update(window)
        q_canon = robot.to_canonical(sim.data.qpos[sim.qadr])
        joint_err.append(float(np.sqrt(np.mean((q_canon - window[0, :29]) ** 2))))
        last_ref = np.asarray(window[0], dtype=np.float64)
      except zmq.Again:
        pass
      if trace is not None:
        rows["step"].append(step)
        rows["q"].append(robot.to_canonical(sim.data.qpos[sim.qadr]))
        rows["quat"].append(sim.measure()[0])
        rows["ref"].append(last_ref)
        rows["qpos"].append(sim.data.qpos.copy())
      total += 1
      if fallen(sim):
        falls += 1
        print(f"[sim_node] fall {falls} at {total * robot.control_dt:.1f} s: back to the stand",
              flush=True)
        sim.reset(init, np.zeros(sim.model.nv))
        pending.clear()
        hold = sim.data.qpos[sim.qadr].copy()
        mujoco.mj_forward(sim.model, sim.data)
        start = _planar_pose(sim)
        step = 0
      else:
        step += 1
      if recorder:
        recorder.capture(total)
      if live:
        if not live.viewer.is_running():
          break
        live.show(total)
      time.sleep(max(0.0, robot.control_dt - (time.monotonic() - t0)))
  except KeyboardInterrupt:
    pass
  finally:
    action_in.close()
    display_in.close()
    state_out.close()
  (x0, y0, yaw0), (x1, y1, yaw1) = start, _planar_pose(sim)
  c, s = math.cos(yaw0), math.sin(yaw0)
  summary = {"robot": robot.name, "mode": "teleop", "steps": total,
             "seconds": round(time.monotonic() - t_start, 1), "falls": falls,
             # Since the start (or the last fall), in the start heading frame.
             "travel_along_m": (x1 - x0) * c + (y1 - y0) * s,
             "travel_across_m": -(x1 - x0) * s + (y1 - y0) * c,
             "turn_deg": math.degrees(math.remainder(yaw1 - yaw0, 2 * math.pi)),
             "joint_err_rms_rad": float(np.mean(joint_err)) if joint_err else None,
             "joint_err_p95_rad": float(np.percentile(joint_err, 95)) if joint_err else None,
             "delay_ms": delay_ms}
  if recorder:
    recorder.save(video, 1.0 / robot.control_dt)
    summary["video"] = str(video)
  if trace is not None:
    np.savez_compressed(trace, step=np.asarray(rows["step"]), q=np.asarray(rows["q"]),
                        quat=np.asarray(rows["quat"]), ref=np.asarray(rows["ref"]),
                        qpos=np.asarray(rows["qpos"]), control_dt=robot.control_dt,
                        joint_names=np.asarray([robot.native_joint_names[i]
                                                for i in robot.perm]))
    summary["trace"] = str(trace)
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("xml", type=Path, help="the robot's exported scene XML")
  parser.add_argument("--robot", required=True)
  parser.add_argument("--clip", type=Path, help="the clip to play (not with --teleop)")
  parser.add_argument("--teleop", action="store_true",
                      help="driven live by policy_node.py --source teleop")
  parser.add_argument("--max-steps", type=int, default=None, help="teleop: stop after N steps")
  parser.add_argument("--robot-json", type=Path, default=None,
                      help="robot constants (default: deploy/sim2sim/robots/<robot>.json)")
  parser.add_argument("--delay-ms", type=float, default=0.0,
                      help="actuator delay; training randomised it over 0-40 ms")
  parser.add_argument("--video", type=Path, default=None)
  parser.add_argument("--realtime", action="store_true", help="pace the loop at 50 Hz")
  parser.add_argument("--viewer", choices=["none", "native"], default="none",
                      help="native: MuJoCo's live viewer (needs a display; real time)")
  parser.add_argument("--overlay", choices=["ghost"], default="ghost",
                      help="drawn in the viewer and video: the reference ghost")
  parser.add_argument("--summary", type=Path, default=None, help="write the summary json")
  args = parser.parse_args()
  if args.robot_json:
    robot = Robot.from_dict(json.loads(args.robot_json.read_text()))
  else:
    robot = Robot.load(args.robot)
  if args.teleop:
    summary = run_teleop(args.xml, robot, args.delay_ms, args.video,
                         viewer=args.viewer == "native", max_steps=args.max_steps)
  elif args.clip is None:
    parser.error("give --clip, or --teleop")
  else:
    summary = run(args.xml, robot, args.clip, args.delay_ms, args.video,
                  realtime=args.realtime, viewer=args.viewer == "native", overlay=args.overlay)
  text = json.dumps(summary, indent=1)
  print(text)
  if args.summary:
    args.summary.write_text(text + "\n")


if __name__ == "__main__":
  main()
