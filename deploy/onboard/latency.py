"""The robot side of the policy protocol at 50 Hz with a fixed standing state, no robot:
how long each step's action takes to come back from a running policy node. Run it beside
the policy node (and the streamer, for teleop), never during a hardware run: it binds the
action port.

  .venv/bin/python -m deploy.onboard.latency deploy/ckpt/<name> [--robot g1] [--seconds 30]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

from deploy.hardware.node import stand_pose
from deploy.sim2sim import protocol
from deploy.sim2sim.obs import Robot


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("deploy", type=Path, help="the export folder the policy node runs")
  parser.add_argument("--robot", default="g1", help="the robot the policy node runs")
  parser.add_argument("--seconds", type=float, default=30.0)
  args = parser.parse_args()

  robot = Robot.from_dict(json.loads((args.deploy / f"{args.robot}.json").read_text())["robot"])
  q = stand_pose(robot, "neutral")
  action_in = protocol.receiver(protocol.ACTION_PORT, protocol.HOST, timeout=0.0)
  state_out = protocol.sender(protocol.STATE_PORT, protocol.HOST, timeout=0.002)
  time.sleep(1.0)
  latency, next_tick = [], time.perf_counter()
  for step in range(round(args.seconds / robot.control_dt)):
    delay = next_tick - time.perf_counter()
    if delay > 0:
      time.sleep(delay)
    next_tick += robot.control_dt
    start = time.perf_counter()
    state_out.send(protocol.pack_state(step, np.array([1.0, 0, 0, 0]), np.zeros(3), q,
                                       np.zeros(robot.num_native), 0))
    latency.append(_await(action_in, step, start + 0.019, robot.num_native) - start)
  # The first second: the policy node clears its history and warms up.
  ms = np.array(latency[round(1 / robot.control_dt):]) * 1e3
  got = ms[np.isfinite(ms)]
  if not len(got):
    raise SystemExit("no action came back: is the policy node running on this export?")
  print(f"{len(ms)} steps: median {np.median(got):.2f} ms, p99 {np.percentile(got, 99):.2f}, "
        f"max {got.max():.2f}; late at 10 ms {np.mean(~(ms <= 10)) * 100:.1f} %, "
        f"at 15 ms {np.mean(~(ms <= 15)) * 100:.1f} %")


def _await(sock: zmq.Socket, step: int, deadline: float, n: int) -> float:
  """When the action for `step` arrived (perf_counter), NaN if not by `deadline`."""
  while (remaining := deadline - time.perf_counter()) > 0:
    sock.setsockopt(zmq.RCVTIMEO, max(1, round(remaining * 1e3)))
    try:
      got, _ = protocol.unpack_action(sock.recv(), n)
    except zmq.Again:
      return np.nan
    if got == step:
      return time.perf_counter()
  return np.nan


if __name__ == "__main__":
  main()
