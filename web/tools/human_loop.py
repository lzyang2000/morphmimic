"""The live demo's loop on the human interface, in Python: the reference the JS port
(src/live/reference.js `LoopReference`'s human track) is checked against, and a runner that
writes the Python figures tools/headless_live.mjs compares the JS loop with.

  uv run --group export python web/tools/human_loop.py [--robots g1 x2 t2] [--points full five]

`deploy.run` plays no clip from the stand on the human interface (`--from-stand` leads into a
robot clip), so the demo's loop (stand, lead-in, clip, return, hold) has its human version
here: `HumanCuedClipReference` is `policy_node.CuedClipReference` on human frames [T, 168]
(a clip's human sidecar, frame k with the robot clip's frame k), built from the deploy code's
human operations (`human_stand`, `human_heading`, `turn_human`, `blend_human`). It takes the
same cues as the robot clip's `CuedClipReference`, so the two play in step: the robot one is
the reference the joint error is measured against, as `sim_node.run_teleop` measures it.

The runner plays every live clip once per robot and slot setting as `deploy.run --clip <c>
--from-stand 2 --cue-s 1` does on the motion interface: the robot in the neutral stand, the cue
at 1 s, until 2 s after the motion; a fall resets the robot to the stand and the step count to
0. It writes `.cache/headless_live/py/<robot>_<full|five>_<clip>.json` (joint_err_rms_rad, falls,
travel_along_m, travel_across_m).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
if str(REPO) not in sys.path:  # Run as a script: `deploy` and `web` live at the repo root.
  sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from deploy.sim2sim.human import (  # noqa: E402
  blend_human,
  human_heading,
  human_stand,
  join_frames,
  turn_human,
)
from deploy.sim2sim.obs import GOAL_K, reference_window  # noqa: E402
from deploy.sim2sim.policy_node import (  # noqa: E402
  FPS,
  REFERENCE_LEAD,
  CuedClipReference,
)
from deploy.vr_teleop.constants import STAND_BLEND_S  # noqa: E402

HUMAN_DIR = WEB / ".cache" / "live" / "human"
POINTS = ("full", "five")


def human_frames(clip: str) -> np.ndarray:
  """A live clip's human sidecar as human frames [T, 168] (policy_node.human_clip)."""
  d = np.load(HUMAN_DIR / f"{clip}.npz")
  return join_frames(d["human_pos_w"].astype(np.float64), d["human_quat_w"].astype(np.float64))


def placed_human(frames: np.ndarray, yaw: float, at: np.ndarray) -> np.ndarray:
  """Human frames [T, 168] turned and moved so that frame 0's pelvis faces `yaw` (rad, the
  robot's convention: `human_heading`) at `at` (x, y); policy_node.placed for human frames."""
  out = turn_human(frames, yaw - human_heading(frames[0]), frames[0, :2])
  pos = out.reshape(len(out), -1, 7)
  pos[..., :2] += at
  return pos.reshape(len(out), -1)


class HumanCuedClipReference(CuedClipReference):
  """`CuedClipReference` on human frames: the uniform skeleton's neutral stand facing the
  robot, then on a cue `lead_in_s` from the stand into the clip's frame 0 (turned to the
  robot's slot-0 heading, placed where the stand is), the clip, `return_s` back into the
  stand placed where the clip ends, and the stand again. `states` holds the human frames."""

  def __init__(self, frames: np.ndarray, robot, lead_in_s: float = 2.0,
               return_s: float = STAND_BLEND_S) -> None:
    self.states = np.asarray(frames, np.float64)
    self.robot = robot
    self.lead_in, self.ret = round(lead_in_s * FPS), round(return_s * FPS)
    self.stand = human_stand()
    self.reset()

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    from deploy.sim2sim.teleop import heading

    if self._idle is None:
      yaw = heading(self.robot.slot0_quat(imu_quat, q))
      self._idle = np.repeat(placed_human(self.stand[None], yaw, np.zeros(2)), GOAL_K, 0)
    if self._pending:
      self._pending = False
      self._played = self._motion(heading(self.robot.slot0_quat(imu_quat, q)))
      self._start = step_id
    if self._played is not None:
      frame = step_id - self._start + REFERENCE_LEAD
      if frame < len(self._played):
        return reference_window(self._played, frame)
      self._idle = np.repeat(self._played[-1:], GOAL_K, 0)
      self._played = None
    return self._idle

  def _motion(self, yaw: float) -> np.ndarray:
    start = self._idle[0]
    clip = placed_human(self.states, yaw, start[:2])
    end = clip[-1]
    stand = placed_human(self.stand[None], human_heading(end), end[:2])[0]
    lead = [blend_human(start, clip[0], (i + 1) / self.lead_in) for i in range(self.lead_in)]
    back = [blend_human(end, stand, (i + 1) / self.ret) for i in range(self.ret)]
    return np.concatenate([np.reshape(x, (-1, clip.shape[1])) for x in (lead, clip, back)])


def run_clip(name: str, points: str, clip: str, deploy: Path) -> dict:
  """One clip from the stand on the human interface with `points`, as deploy.run --from-stand
  2 --cue-s 1 runs the motion interface: the summary deploy.run writes, the parts compared."""
  from deploy.sim2sim import sim_node
  from deploy.sim2sim.obs import Robot
  from deploy.sim2sim.policy_node import PolicyRunner
  from deploy.vr_teleop.gmr_retarget import STAND_FILE
  from web.tools.gen_live_assets import POLICY_ROBOT

  doc = json.loads((deploy / f"{POLICY_ROBOT.get(name, name)}.json").read_text())
  robot = Robot.from_dict(doc["robot"])
  states = np.load(WEB / ".cache" / "live" / "clips" / name / f"{clip}.npz")["states"]
  motion = CuedClipReference(states, robot, 2.0)
  human = HumanCuedClipReference(human_frames(clip), robot, 2.0)
  runner = PolicyRunner(deploy, robot.name, human, "human", points)
  sim = sim_node.SimRobot(WEB / ".cache" / "assets" / f"{robot.name}.xml", robot)
  init = sim_node.stand_qpos(sim.model, sim, json.loads(STAND_FILE.read_text())[robot.body])
  sim.reset(init, np.zeros(sim.model.nv))
  cue = round(1.0 * FPS)
  total = cue + len(motion) + 2 * FPS
  start = sim_node._planar_pose(sim)
  errs, falls, step, prev = [], 0, 0, -1
  for _ in range(total):
    quat, gyro, q, dq = sim.measure()
    cues = int(step >= cue)
    targets = runner.step(step, quat, gyro, q, dq, cues)  # resets the human reference itself
    if step < prev:
      motion.reset()
    prev = step
    motion.cue(step, cues)
    window = motion.window(step, quat / np.linalg.norm(quat), q)
    for _ in range(robot.decimation):
      sim.set_targets(targets)
      sim.step()
    mujoco.mj_forward(sim.model, sim.data)
    q_canon = robot.to_canonical(sim.data.qpos[sim.qadr])
    errs.append(float(np.sqrt(np.mean((q_canon - window[0, :29]) ** 2))))
    if sim_node.fallen(sim):
      falls += 1
      sim.reset(init, np.zeros(sim.model.nv))
      mujoco.mj_forward(sim.model, sim.data)
      start = sim_node._planar_pose(sim)
      step = 0
    else:
      step += 1
  (x0, y0, yaw0), (x1, y1, _) = start, sim_node._planar_pose(sim)
  c, s = np.cos(yaw0), np.sin(yaw0)
  return {"robot": name, "clip": clip, "interface": f"human {points}", "falls": falls,
          "joint_err_rms_rad": float(np.mean(errs)),
          "travel_along_m": float((x1 - x0) * c + (y1 - y0) * s),
          "travel_across_m": float(-(x1 - x0) * s + (y1 - y0) * c)}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--robots", nargs="+", default=["g1", "x2", "t2"])
  parser.add_argument("--points", nargs="+", default=list(POINTS), choices=POINTS)
  args = parser.parse_args()
  from web.tools.xcheck_live import deploy_dir

  deploy = deploy_dir()
  out = WEB / ".cache" / "headless_live" / "py"
  out.mkdir(parents=True, exist_ok=True)
  clips = [c["id"] for c in json.loads((WEB / "public" / "live" / "clips.json").read_text())["clips"]]
  for name in args.robots:
    for points in args.points:
      for clip in clips:
        s = run_clip(name, points, clip, deploy)
        (out / f"{name}_{points}_{clip}.json").write_text(json.dumps(s, indent=1) + "\n")
        print(f"{name} {points} {clip}: falls {s['falls']}, joint err {s['joint_err_rms_rad']:.4f}",
              flush=True)


if __name__ == "__main__":
  main()
