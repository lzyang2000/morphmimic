"""A deployment in one command, in the sim or on the robot: `sim2sim/policy_node.py`
as its own process, and here the robot side, `sim2sim/sim_node.py` (MuJoCo) or with
`--hardware g1` the real G1 (`hardware/g1_node.py`). The two speak only the `sim2sim/protocol.py` messages, so the policy
process cannot tell them apart.

  uv run --group export python -m deploy.run --deploy <deploy_dir> \\
    --assets <assets_dir> --robot g1 --clip walk_hands_on_back_loop_002__A096 \\
    [--delay-ms 20] [--video out.mp4] [--viewer native]

Teleop (`--teleop`): the policy node follows the GMR streamer instead of a clip. Run the
streamer yourself (`python -m deploy.vr_teleop.streamer --robot <robot>`, the headset), or
give `--replay F` to start it here on a PICO recording:

  uv run --group export --group teleop python -m deploy.run --deploy <deploy_dir> \\
    --assets <assets_dir> --robot x2 --teleop [--replay session.npz] [--viewer native]

Neither `--clip` nor `--teleop`: the policy tracks the neutral stand (`stand.json`); the
sim starts the robot on it, and `--max-steps` ends the run (sim) or the policy loop
(hardware, then a PD hold on the stand). The first hardware run, 30 s:

  uv run --group export --group teleop python -m deploy.run --deploy <deploy_dir> \
    --robot g1 --hardware g1 --net enp3s0 --max-steps 1500

Hardware (`--hardware g1 --net IFACE`): the real G1 (`deploy.hardware.g1_node`, its
remote's START / A / B / SELECT, or with `--pad xbox|ps` a joystick beside it) takes the
sim's place, with the same policy node (a clip or teleop). `--assets` is then needed only
for a clip. A clip on hardware starts from the
stand: after A the policy holds the neutral stand, and each Y on the remote plays the
clip, turned to the robot's heading and led into over `--from-stand` s (default 2), then
back to the stand. `--from-stand` in the sim does the same from the stand, facing
`--start-yaw`, Y pressed at `--cue-s` (default 1 s; the run ends 2 s after the clip):

  uv run --group export --group teleop python -m deploy.run --deploy <deploy_dir> \
    --assets <assets_dir> --robot g1 --clip walk_forward_start_001__A021 --from-stand 2 \
    --start-yaw 137 --cue-s 1 15 --viewer native

  uv run --group export --group teleop python -m deploy.run --deploy <deploy_dir> \\
    --robot g1 --hardware g1 --net enp3s0 --teleop

Human interface (`--interface human --human-points full|five`): the policy acts through
the human skeleton goal (`action_human`) instead of the robot reference. In teleop the
streamer runs `--target human` (PICO on the uniform skeleton, no GMR); a clip needs its
human sidecar too (`--human-clip <human50>/<date>/<stem>.npz`; the sim starts the robot on
the robot clip and scores it against it):

  uv run --group export --group teleop python -m deploy.run --deploy <deploy_dir> \\
    --assets <assets_dir> --robot g1 --interface human --human-points five --teleop \\
    --replay session.npz

A fall (`--tilt-deg`, `--stand-tilt-deg`) damps the robot.

`<deploy_dir>` is an `export_onnx.py` folder (policy.onnx, <robot>.json); `<assets_dir>`
is an `export_assets.py` folder (<robot>.xml, clips/<robot>/<clip>.npz). The two
processes talk only through the `protocol` ZMQ messages, as on the robot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from deploy.hardware.node import Recording
from deploy.hardware.pad import LAYOUTS, Pad
from deploy.sim2sim import sim_node
from deploy.sim2sim.obs import BODIES, Robot
from deploy.vr_teleop.constants import STAND_BLEND_S


def _forward(proc: subprocess.Popen) -> None:
  """Print a child's output from a thread, so a long session never fills the pipe."""
  threading.Thread(target=lambda: [print(x, end="", flush=True) for x in proc.stdout],
                   daemon=True).start()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--deploy", type=Path, required=True)
  parser.add_argument("--assets", type=Path, default=None,
                      help="export_assets.py folder (the sim's scene; the clips)")
  parser.add_argument("--robot", required=True)
  parser.add_argument("--clip", help="clip name (assets/clips/<robot>/); not with --teleop"
                      " (neither: the policy stands)")
  parser.add_argument("--teleop", action="store_true", help="follow the streamer")
  parser.add_argument("--interface", choices=["motion", "human"], default="motion",
                      help="the goal the policy acts through (policy_node.py --interface)")
  parser.add_argument("--human-points", choices=["full", "five", "six", "three"],
                      default="full", help="--interface human: the slot setting")
  parser.add_argument("--human-clip", type=Path, default=None,
                      help="--interface human with --clip: the clip's human sidecar")
  parser.add_argument("--replay", type=Path, default=None,
                      help="teleop: start the streamer here on this PICO recording")
  parser.add_argument("--streamer", default=None,
                      help="teleop: the streamer's endpoint (default tcp://127.0.0.1:5556)")
  parser.add_argument("--max-steps", type=int, default=None,
                      help="teleop, stand: stop after N steps (hardware: per A press)")
  parser.add_argument("--delay-ms", type=float, default=0.0,
                      help="sim: actuator delay; hardware: each policy command goes out this"
                      " long after its state was read (at least)")
  parser.add_argument("--video", type=Path, default=None)
  parser.add_argument("--viewer", choices=["none", "native"], default="none",
                      help="native: MuJoCo's live viewer (needs a display; real time)")
  parser.add_argument("--summary", type=Path, default=None)
  parser.add_argument("--hardware", choices=["g1"], default=None,
                      help="drive the real robot instead of the sim (deploy.hardware.g1_node)")
  parser.add_argument("--net", default=None, help="hardware: network interface to the robot")
  parser.add_argument("--kp-scale", type=float, default=1.0, help="hardware: stiffness scale")
  parser.add_argument("--stand", choices=["neutral", "default"], default="neutral",
                      help="hardware: the pose START stands the robot in")
  parser.add_argument("--record", type=Path, default=None, help="hardware: save every tick")
  parser.add_argument("--pad", choices=list(LAYOUTS), default=None,
                      help="hardware: a joystick as a second button source, with this layout")
  parser.add_argument("--pad-device", default="/dev/input/js0", help="hardware: the joystick")
  parser.add_argument("--tilt-deg", type=float, default=80.0,
                      help="hardware: damp when the pelvis tilts more than this from upright"
                      " (deg) while the policy or a blend to the stand drives the robot")
  parser.add_argument("--stand-tilt-deg", type=float, default=30.0,
                      help="hardware: the same once the target is the stand itself (deg)")
  parser.add_argument("--from-stand", type=float, default=None, metavar="S",
                      help="clip: start on the stand, lead into the clip over S s"
                      " (hardware: always, default 2)")
  parser.add_argument("--start-yaw", type=float, default=0.0,
                      help="sim, from the stand: the robot's heading at the start (deg)")
  parser.add_argument("--cue-s", type=float, nargs="+", default=[1.0],
                      help="sim, from the stand: when Y is pressed (s)")
  parser.add_argument("--trace", type=Path, default=None,
                      help="sim, from the stand: write the robot's joints, IMU and the tracked"
                      " reference per step (npz)")
  args = parser.parse_args()
  if args.teleop and args.clip is not None:
    parser.error("give --clip or --teleop, not both")
  standing = not args.teleop and args.clip is None
  if args.hardware and (args.net is None or BODIES.get(args.robot, args.robot) != args.hardware):
    parser.error("--hardware needs --net, and --robot set to the same robot")
  if args.assets is None and (args.clip is not None or not args.hardware):
    parser.error("--assets is needed for a clip and for the sim")
  human = args.interface == "human"
  if human and args.clip is not None and args.human_clip is None:
    parser.error("--interface human with --clip needs --human-clip")
  if human and (args.from_stand is not None or (args.hardware and args.clip is not None)):
    parser.error("--interface human plays no clip from the stand (--from-stand, a clip on"
                 " hardware): use --teleop or the stand")
  if args.hardware and args.clip is not None and args.from_stand is None:
    args.from_stand = 2.0

  robot = Robot.from_dict(json.loads((args.deploy / f"{args.robot}.json").read_text())["robot"])
  policy_cmd = [sys.executable, "-m", "deploy.sim2sim.policy_node", str(args.deploy),
                "--robot", args.robot, "--idle-exit-s", "0", "--interface", args.interface,
                "--human-points", args.human_points]
  if args.teleop:
    policy_cmd += ["--source", "teleop"] + (["--streamer", args.streamer] if args.streamer else [])
  elif standing:
    policy_cmd += ["--source", "stand"]
  else:
    clip = args.assets / "clips" / args.robot / f"{args.clip}.npz"
    policy_cmd += (["--source", "clip", "--human-clip", str(args.human_clip)] if human
                   else ["--clip", str(clip)])
    if args.from_stand is not None:
      policy_cmd += ["--from-stand", str(args.from_stand)]
      if args.max_steps is None and not args.hardware:  # 2 s after the last cue's motion.
        played = (round(args.from_stand / robot.control_dt) + len(np.load(clip)["states"])
                  + round(STAND_BLEND_S / robot.control_dt))
        args.max_steps = round(max(args.cue_s) / robot.control_dt) + played + 100
  children = []
  try:
    if args.replay:
      target = ["--target", "human"] if human else ["--robot", BODIES.get(args.robot, args.robot)]
      streamer = subprocess.Popen(
        [sys.executable, "-m", "deploy.vr_teleop.streamer", *target, "--replay",
         str(args.replay)], stdout=subprocess.PIPE, text=True)
      children.append(streamer)
      for line in streamer.stdout:  # Up to its ready line (GMR takes a few s to load).
        print(line, end="", flush=True)
        if "publishing on" in line:
          break
      else:
        raise RuntimeError("the streamer exited before it was ready")
      _forward(streamer)
    policy = subprocess.Popen(policy_cmd, stdout=subprocess.PIPE, text=True)
    children.append(policy)
    line = policy.stdout.readline()  # the ready line: the session is loaded
    if not line:
      raise RuntimeError("policy_node.py exited before it was ready")
    print(line, end="", flush=True)
    _forward(policy)
    viewer = args.viewer == "native"
    if args.hardware:
      import signal

      def on_term(*_) -> None:
        """SIGTERM stops like Ctrl-C (the node damps on the way out); repeats are ignored
        so they cannot interrupt the damping."""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

      signal.signal(signal.SIGTERM, on_term)
      rec = Recording() if args.record else None
      pad = Pad.open(args.pad_device, args.pad) if args.pad else None
      try:
        from deploy.hardware import g1_node

        node = g1_node.make_node(args.deploy, args.net, args.stand, args.kp_scale, args.robot,
                                 send_delay_s=args.delay_ms / 1e3, max_steps=args.max_steps, record=rec,
                                 buttons=(pad,) if pad is not None else (),
                                 tilt_deg=args.tilt_deg, stand_tilt_deg=args.stand_tilt_deg)
        try:
          node.run()
        finally:
          node.close()
      finally:
        if pad is not None:
          pad.close()
        if rec is not None and rec.rows["t"]:
          print(f"[run] saved {rec.save(args.record)} ticks to {args.record}", flush=True)
      summary = {"robot": args.robot, "mode": "hardware"}
    elif args.teleop or standing or args.from_stand is not None:
      xml = args.assets / f"{args.robot}.xml"
      summary = sim_node.run_teleop(xml, robot, args.delay_ms, args.video, viewer=viewer,
                                    max_steps=args.max_steps, start_yaw_deg=args.start_yaw,
                                    cue_steps=tuple(round(t / robot.control_dt)
                                                    for t in args.cue_s),
                                    trace=args.trace)
      summary["mode"] = ("stand" if standing else "teleop" if args.teleop
                         else f"clip {args.clip} from the stand")
    else:
      xml = args.assets / f"{args.robot}.xml"
      summary = sim_node.run(xml, robot, clip, args.delay_ms, args.video, viewer=viewer)
  finally:
    for child in reversed(children):
      child.terminate()
      child.wait(timeout=10)
  summary["interface"] = f"human {args.human_points}" if human else "motion"
  text = json.dumps(summary, indent=1)
  print(text)
  if args.summary:
    args.summary.write_text(text + "\n")


if __name__ == "__main__":
  main()
