"""Read-only check of the G1 from its own computer, before a run: Unitree's motion mode,
the pelvis IMU, every joint encoder with its motor temperature and torque, and the
remote. It subscribes only (the SDK's command writer stays silent until a command is
set), so it is safe in any state the robot is in.

  .venv/bin/python -m deploy.onboard.g1.check_link [--net eth0]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.sim2sim.obs import Robot


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--net", default="eth0", help="the robot's network interface (DDS)")
  args = parser.parse_args()
  import unitree_interface

  names = Robot.load("g1").native_joint_names
  sdk = unitree_interface.UnitreeInterface.create_g1(args.net)
  time.sleep(1.0)  # The first messages.
  mode = sdk.check_motion_mode()
  print(f"motion mode {mode!r}: " + ("released, low-level commands take effect" if not mode
                                     else "Unitree's controller; g1_node releases it on start"))
  samples = []
  for _ in range(50):
    samples.append(sdk.read_low_state())
    time.sleep(0.02)
  state = samples[-1]
  quat = np.asarray(state.imu.quat, float)
  if not np.linalg.norm(quat) > 0.5:
    raise SystemExit(f"no low state on {args.net} (robot off, or the wrong interface)")
  qs = np.array([s.motor.q[:len(names)] for s in samples])
  print(f"low state live: {len(np.unique(qs.round(6), axis=0))} distinct readings of 50 in 1 s")
  w, x, y, z = quat
  print(f"pelvis tilt {np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))):.1f} deg, "
        f"gyro {np.round(state.imu.omega, 3)} rad/s")
  print(f"{'joint':28s} {'q rad':>7s} {'q deg':>7s} {'temp C':>7s} {'tau Nm':>7s}")
  for i, name in enumerate(names):
    print(f"{name:28s} {state.motor.q[i]:7.3f} {np.degrees(state.motor.q[i]):7.1f} "
          f"{state.motor.temperature[i]:7.0f} {state.motor.tau_est[i]:7.2f}")
  ctrl = sdk.read_wireless_controller()
  print(f"remote: keys {ctrl.keys:#06x}, sticks lx {ctrl.lx:.2f} ly {ctrl.ly:.2f} "
        f"rx {ctrl.rx:.2f} ry {ctrl.ry:.2f}")


if __name__ == "__main__":
  main()
