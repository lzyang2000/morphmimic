"""The G1 on hardware: the robot side of the sim2sim protocol (`humanoid_dodgeball`'s
hardware node, speaking our `deploy/sim2sim/protocol.py`).

It owns the Unitree SDK (`unitree_interface`, built from `third_party/unitree_sdk2_wrapper`
by `deploy/install_unitree_sdk.sh`): it reads the IMU (the pelvis IMU, the G1's
`Robot.imu_body`: quaternion wxyz and body-frame gyro), the 29 joint encoders and the
wireless remote, and writes 29 PD targets at 50 Hz. The state machine and the policy loop
are `HardwareNode`'s (`node.py`); the policy node (`policy_node.py`, clip or teleop) does
everything else, unchanged from the sim: per step this node sends the state message and
applies the action of the same `step_id`. It has no pose source yet (`pose_source.py`):
its state messages carry a NaN pose, and it refuses an export of the robot goal anchor
(`--goal-anchor robot`).

The gains are the ones the policy was trained with, from the deployment folder's
`<robot>.json` (the robot constants' `pd`: mjlab's G1 actuators). They differ from
`humanoid_dodgeball`'s on the hip pitch and the wrist pitch/yaw, so never copy gains
across projects. `--kp-scale` softens the stiffness for a cautious first run.

The physical remote has the last word, as in `humanoid_dodgeball`:

  START   from damping, interpolate over 2 s to the stand (`--stand`: the neutral stand
          of `stand.json`, default, or the training default pose) and hold it
  A       start the policy loop (the step count restarts at 0: the policy node clears
          its history; in teleop it then holds OFF until a recalibration; with a clip,
          or without one, the policy holds the neutral stand, aimed at the robot's heading)
  Y       during the policy loop: play the clip (from the stand, turned to the robot's
          heading, then back to the stand; again on the next Y)
  B       graceful stop: damp (kp 0, kd only) at the current pose; START stands again
  SELECT  emergency stop: damp and exit

`--pad xbox|ps` adds a joystick (`pad.py`; `--pad-device`, default `/dev/input/js0`) as a
second source of the same five buttons, START, A, Y, B and SELECT of its layout; the
remote keeps working beside it.

The loop waits up to `--action-wait-ms` for this step's action (else it holds the last
target), and if no new action arrives for `--action-timeout-s` it restarts the step count
and blends to the PD stand until the policy answers. With `--max-steps N` the policy runs
N steps (50 per second), then the node blends over 2 s back to the PD stand and holds it:
A runs the policy again, B damps (the details are in `node.py`).

The safety layer is `node.py`'s, as on the X2: a fall (the pelvis more than `--tilt-deg`,
default 80, from upright while the policy or a blend drives it, more than
`--stand-tilt-deg`, default 30, once the target is the stand) damps in every state that
commands the robot, an invalid state (non-finite, or a near-zero quaternion) goes to the
PD stand and refuses START. The SDK reports no input ages and no motor faults, so the G1
adds no checks of its own.

  uv run --group export python -m deploy.hardware.g1_node --deploy <deploy_dir> \\
    --net <iface> [--kp-scale 1.0] [--max-steps 1500] [--policy-host 127.0.0.1] \\
    [--pad xbox] [--record ticks.npz]

Its logic runs against any object with the SDK's methods (`G1Node(sdk, ...)`), so it
can be tested with a fake SDK, without a robot.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np

from deploy.hardware.node import Gains, HardwareNode, Recording, Sample, stand_pose
from deploy.hardware.pad import LAYOUTS, Pad
from deploy.hardware.pose_source import NO_POSE_SOURCE
from deploy.sim2sim import protocol
from deploy.sim2sim.obs import Robot

# The remote's `keys` bitmask (the wrapper's WirelessController; its .pyi A/B bools are
# stale, as `humanoid_dodgeball` notes).
KEYS = {"R1": 0x0001, "L1": 0x0002, "start": 0x0004, "select": 0x0008, "R2": 0x0010,
        "L2": 0x0020, "F1": 0x0040, "F2": 0x0080, "A": 0x0100, "B": 0x0200, "X": 0x0400,
        "Y": 0x0800, "up": 0x1000, "right": 0x2000, "down": 0x4000, "left": 0x8000}


def pressed(ctrl, name: str) -> bool:
  return bool(int(getattr(ctrl, "keys", 0)) & KEYS[name])


class G1Node(HardwareNode):
  """The node over a Unitree SDK object: its IMU, encoders, wireless remote and LowCmd."""

  tag = "g1_node"

  def __init__(self, sdk, robot: Robot, gains: Gains, stand_q: np.ndarray, **kwargs):
    super().__init__(robot, gains, stand_q, **kwargs)
    self.sdk = sdk

  def read_state(self) -> Sample:
    state = self.sdk.read_low_state()
    return Sample(quat=np.asarray(state.imu.quat, float),  # wxyz
                  gyro=np.asarray(state.imu.omega, float),  # body frame
                  q=np.asarray(state.motor.q[: self.n], float),
                  dq=np.asarray(state.motor.dq[: self.n], float))

  def send(self, q_target: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
    cmd = self.sdk.create_zero_command()
    cmd.q_target = [float(x) for x in q_target]
    cmd.dq_target = [0.0] * self.n
    cmd.kp = [float(x) for x in kp]
    cmd.kd = [float(x) for x in kd]
    cmd.tau_ff = [0.0] * self.n
    self.sdk.write_low_command(cmd)

  def buttons_held(self) -> set[str]:
    """The remote's buttons (one read of its state) and the pad's, if any."""
    ctrl = self.sdk.read_wireless_controller()
    return {n for n in KEYS if pressed(ctrl, n)} | super().buttons_held()


def make_node(deploy_dir: Path, net: str, stand: str = "neutral", kp_scale: float = 1.0,
              robot_name: str = "g1", **kwargs) -> G1Node:
  """The node on the real G1 (needs `unitree_interface`), with the gains of the
  export's `<robot_name>.json` (a robot on the G1 body: `obs.BODIES`). Raises RuntimeError
  for an export of the robot goal anchor, which needs a pose source the G1 does not have
  yet."""
  full = json.loads((Path(deploy_dir) / f"{robot_name}.json").read_text())
  if full.get("goal_anchor", "reference") == "robot":
    raise RuntimeError(NO_POSE_SOURCE)
  import unitree_interface

  doc = full["robot"]
  robot = Robot.from_dict(doc)
  gains = Gains.from_deploy(doc, kp_scale)
  print(f"[g1_node] gains from {deploy_dir}/{robot_name}.json (x{kp_scale} kp): kp "
        f"{gains.kp.min():.1f}..{gains.kp.max():.1f}, kd {gains.kd.min():.2f}..{gains.kd.max():.2f}",
        flush=True)
  sdk = unitree_interface.UnitreeInterface.create_g1(net)
  if not sdk.release_motion_control():
    print("[g1_node] WARNING: could not release high-level motion control; LowCmd may be"
          " ignored", flush=True)
  sdk.set_control_mode(unitree_interface.ControlMode.PR)
  return G1Node(sdk, robot, gains, stand_pose(robot, stand), **kwargs)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--deploy", type=Path, required=True, help="export_onnx.py folder")
  parser.add_argument("--net", required=True, help="network interface to the robot (DDS)")
  parser.add_argument("--stand", choices=["neutral", "default"], default="neutral")
  parser.add_argument("--kp-scale", type=float, default=1.0)
  parser.add_argument("--policy-host", default=protocol.HOST, help="where policy_node runs")
  parser.add_argument("--bind-host", default=protocol.HOST,
                      help="address the action port listens on (0.0.0.0 for a remote policy)")
  parser.add_argument("--action-wait-ms", type=float, default=10.0)
  parser.add_argument("--action-timeout-s", type=float, default=1.0)
  parser.add_argument("--max-steps", type=int, default=None,
                      help="policy steps per A (50 per second), then hold the stand")
  parser.add_argument("--record", type=Path, default=None, help="save every policy tick")
  parser.add_argument("--pad", choices=list(LAYOUTS), default=None,
                      help="a joystick as a second button source, with this button layout")
  parser.add_argument("--pad-device", default="/dev/input/js0", help="the joystick device")
  parser.add_argument("--tilt-deg", type=float, default=80.0,
                      help="damp when the pelvis tilts more than this from upright (deg)"
                      " while the policy or a blend to the stand drives the robot")
  parser.add_argument("--stand-tilt-deg", type=float, default=30.0,
                      help="the same once the target is the stand itself (deg)")
  args = parser.parse_args()
  stopping = [False]

  def on_term(*_):
    if not stopping[0]:  # The first signal stops; repeats cannot interrupt the damping.
      stopping[0] = True
      raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, on_term)
  rec = Recording() if args.record else None
  pad = Pad.open(args.pad_device, args.pad) if args.pad else None
  node = make_node(args.deploy, args.net, args.stand, args.kp_scale,
                   policy_host=args.policy_host, bind_host=args.bind_host,
                   action_wait_s=args.action_wait_ms / 1e3,
                   action_timeout_s=args.action_timeout_s, max_steps=args.max_steps,
                   record=rec, buttons=(pad,) if pad else (), tilt_deg=args.tilt_deg,
                   stand_tilt_deg=args.stand_tilt_deg)
  try:
    node.run()
  finally:
    node.close()
    if pad is not None:
      pad.close()
    if rec is not None and rec.rows["t"]:
      print(f"[g1_node] saved {rec.save(args.record)} ticks to {args.record}", flush=True)


if __name__ == "__main__":
  main()
