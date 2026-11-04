"""Write .cache/xcheck_live/<robot>_<interface>.json: a recorded sim2sim tick sequence with the
exact policy inputs, for tools/xcheck_live.mjs to reproduce in JS.

  uv run --project .. --group export python tools/xcheck_live.py [--ticks N]

Per robot (g1, x2, t2) and interface (motion; full and five: the human interface with that slot
setting, `tools/human_loop.py`'s HumanCuedClipReference on the clips' human sidecars) it runs the
CPU sim2sim from the stand
in one process, as `deploy.run --clip A --from-stand 2 --cue-s 1` does with its two processes:
`sim_node.SimRobot` in the exported scene (.cache/assets/<robot>.xml; the G1's is
g1_amber.xml, gen_live_assets.POLICY_ROBOT) starts in the neutral
stand (`sim_node.stand_qpos`, facing START_YAW_DEG), and each control step its measurements go
to the real `policy_node.PolicyRunner` with a `CuedClipReference` (lead in 2 s), whose targets
drive the physics for `decimation` steps; a fall resets the robot to the stand and the step
count to 0, as `sim_node.run_teleop` does. The policy is public/live/policy.onnx with the
export's <robot>.json (.cache/live/policy).

The cues follow the live demo's loop (robots.json `loop`): the first at 1 s, the next one
`holdS` after the motion of the previous one has ended (it is back on the stand), as in
src/live/reference.js `LoopReference`; here they are fixed in advance and pressed through the
policy node's cue counter. At tick SWITCH_TICK, mid-motion, the reference's clip is changed to
CLIP_B: the next cue plays it. The run ends AFTER_CUE ticks into the second cue.

Per tick the file holds what the robot measured (step id, IMU quaternion and gyro, q, dq, all
float64), the cue count, the action the policy returned (it is the next tick's last action)
and the native joint targets made from it,
the reference phase, and every input the ONNX was fed except the constant normalisers:
obs_history and goal_motion, or on the human interface goal_human, human_point_mask,
human_frame_mask, human_heading and human_heading_mask. With them: on the motion interface
the five reference points of the window the policy tracked (pelvis, ankles, wrists [K, 5, 3],
`kp`), on the human interface the 24 human slot positions of its window ([K, 24, 3], `human`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
if str(REPO) not in sys.path:  # Run as a script: `deploy` lives at the repo root.
  sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from deploy.sim2sim import sim_node  # noqa: E402
from deploy.sim2sim.obs import Robot  # noqa: E402
from deploy.sim2sim.policy_node import CuedClipReference, PolicyRunner  # noqa: E402
from deploy.vr_teleop.constants import STAND_BLEND_S  # noqa: E402
from deploy.vr_teleop.gmr_retarget import STAND_FILE  # noqa: E402
from web.tools.gen_live_assets import POLICY_ROBOT  # noqa: E402
from web.tools.human_loop import HumanCuedClipReference, human_frames  # noqa: E402

ROBOTS = ("g1", "x2", "t2")
INTERFACES = ("motion", "full", "five")
# The window's five reference points: pelvis, left/right ankle, left/right wrist (obs.js KP_BODY_IDX).
KP_BODY_IDX = (0, 3, 6, 10, 13)
CLIP_A = "dance_basic_padeburee_270_R_loop_001__A324"
CLIP_B = "victory_dance_robot_dance_180_R_003__A306"
LEAD_IN_S = 2.0  # --from-stand 2
FIRST_CUE_S = 1.0  # --cue-s 1
START_YAW_DEG = {"motion": 0.0, "full": 137.0, "five": -64.0}
SWITCH_TICK = 400
AFTER_CUE = 150
OUT = WEB / ".cache" / "xcheck_live"


class RecordingSession:
  """The runner's onnxruntime session, keeping the feeds of the last run."""

  def __init__(self, session) -> None:
    self.session = session
    self.feeds: dict | None = None

  def get_inputs(self):
    return self.session.get_inputs()

  def run(self, names, feeds):
    self.feeds = {k: np.array(v) for k, v in feeds.items()}
    return self.session.run(names, feeds)


def phase(ref: CuedClipReference, step: int) -> str:
  """What the window of `step` was: stand, lead, clip or return (read after `window`)."""
  if ref._played is None:
    return "stand"
  frame = step - ref._start + 1
  clip = len(ref._played) - ref.lead_in - ref.ret
  return "lead" if frame < ref.lead_in else "clip" if frame < ref.lead_in + clip else "return"


def _f32(a: np.ndarray) -> str:
  return "[" + ",".join(np.char.mod("%.9g", np.asarray(a, np.float32).ravel())) + "]"


def _f64(a: np.ndarray) -> str:
  return json.dumps(np.asarray(a, np.float64).ravel().tolist())


def _bools(a: np.ndarray) -> str:
  return json.dumps([int(x) for x in np.asarray(a).ravel()])


def deploy_dir() -> Path:
  """policy.onnx from public/live with the export's <robot>.json (the G1's g1_amber.json), in
  one folder."""
  d = OUT / "deploy"
  d.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(WEB / "public" / "live" / "policy.onnx", d / "policy.onnx")
  for r in ROBOTS:
    r = POLICY_ROBOT.get(r, r)
    shutil.copyfile(WEB / ".cache" / "live" / "policy" / f"{r}.json", d / f"{r}.json")
  return d


def run(name: str, interface: str, deploy: Path, loop: dict, ticks: int | None) -> dict:
  doc = json.loads((deploy / f"{POLICY_ROBOT.get(name, name)}.json").read_text())
  robot = Robot.from_dict(doc["robot"])
  clips = WEB / ".cache" / "live" / "clips" / name
  states_a = np.load(clips / f"{CLIP_A}.npz")["states"]
  states_b = np.load(clips / f"{CLIP_B}.npz")["states"]
  fps = round(1.0 / robot.control_dt)
  human = interface != "motion"
  reference = (HumanCuedClipReference(human_frames(CLIP_A), robot, LEAD_IN_S) if human
               else CuedClipReference(states_a, robot, LEAD_IN_S))
  runner = (PolicyRunner(deploy, robot.name, reference, "human", interface) if human
            else PolicyRunner(deploy, robot.name, reference, interface))
  session = runner.session = RecordingSession(runner.session)

  # The loop's cues: the first at FIRST_CUE_S, the second holdS after the first motion ends
  # (its last window is at step cue + len - 2; the stand is back at cue + len - 1).
  first = round(FIRST_CUE_S * fps)
  played = round(LEAD_IN_S * fps) + len(states_a) + round(STAND_BLEND_S * fps)
  second = first + played - 1 + round(loop["holdS"] * fps)
  cue_steps = (first, second)
  total = ticks if ticks is not None else second + AFTER_CUE

  stand = json.loads(STAND_FILE.read_text())[robot.body]
  sim = sim_node.SimRobot(WEB / ".cache" / "assets" / f"{robot.name}.xml", robot)
  init = sim_node.stand_qpos(sim.model, sim, stand, START_YAW_DEG[interface])
  sim.reset(init, np.zeros(sim.model.nv))
  kp = list(KP_BODY_IDX)
  rows: list[str] = []
  phases: dict[str, int] = {}
  step, falls = 0, 0
  for tick in range(total):
    if tick == SWITCH_TICK:
      # Played from the next cue.
      reference.states = human_frames(CLIP_B) if human else np.asarray(states_b, np.float64)
    quat, gyro, q, dq = sim.measure()
    cues = sum(step >= c for c in cue_steps)
    targets = runner.step(step, quat, gyro, q, dq, cues)
    feeds = session.feeds
    ph = phase(reference, step)
    phases[ph] = phases.get(ph, 0) + 1
    if human:
      points = runner.last_ref.reshape(-1, 24, 7)[..., :3]
      goal = (f'"human":{_f64(points)},"goal_human":{_f32(feeds["goal_human"])},'
              f'"human_point_mask":{_bools(feeds["human_point_mask"])},'
              f'"human_frame_mask":{_bools(feeds["human_frame_mask"])},'
              f'"human_heading":{_f32(feeds["human_heading"])},'
              f'"human_heading_mask":{_bools(feeds["human_heading_mask"])}')
    else:
      window_kp = runner.last_ref[:, 58:100].reshape(-1, 14, 3)[:, kp]
      goal = f'"kp":{_f64(window_kp)},"goal_motion":{_f32(feeds["goal_motion"])}'
    rows.append(
      "{" + f'"tick":{tick},"step":{step},"cues":{cues},"phase":"{ph}",'
      f'"quat":{_f64(quat)},"gyro":{_f64(gyro)},"q":{_f64(q)},"dq":{_f64(dq)},'
      f'"action":{_f32(runner.last_action)},"targets":{_f64(targets)},'
      f'"obs_history":{_f32(feeds["obs_history"])},{goal}' + "}"
    )
    for _ in range(robot.decimation):
      sim.set_targets(targets)
      sim.step()
    mujoco.mj_forward(sim.model, sim.data)
    if sim_node.fallen(sim):
      falls += 1
      sim.reset(init, np.zeros(sim.model.nv))
      step = 0
    else:
      step += 1
  meta = {
    "robot": name, "interface": interface, "startYawDeg": START_YAW_DEG[interface],
    "clipA": CLIP_A, "clipB": CLIP_B, "switchTick": SWITCH_TICK, "firstCueS": FIRST_CUE_S,
    "leadInS": LEAD_IN_S, "returnS": STAND_BLEND_S, "holdS": loop["holdS"],
    "cueSteps": list(cue_steps), "falls": falls, "phases": phases, "ticks": total,
  }
  path = OUT / f"{name}_{interface}.json"
  path.write_text(json.dumps(meta)[:-1] + ',"rows":[\n' + ",\n".join(rows) + "\n]}\n")
  return {**meta, "file": str(path.relative_to(WEB))}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--ticks", type=int, default=None, help="shorter run (debugging)")
  parser.add_argument("--robots", nargs="+", default=list(ROBOTS))
  parser.add_argument("--interfaces", nargs="+", default=list(INTERFACES))
  args = parser.parse_args()
  live = json.loads((WEB / "public" / "live" / "robots.json").read_text())
  loop = live["loop"]
  if (loop["leadInS"], loop["returnS"]) != (LEAD_IN_S, STAND_BLEND_S):
    raise SystemExit(f"robots.json loop {loop} is not lead-in {LEAD_IN_S} s, return {STAND_BLEND_S} s")
  deploy = deploy_dir()
  for name in args.robots:
    for interface in args.interfaces:
      s = run(name, interface, deploy, loop, args.ticks)
      print(f"{name} {interface}: {s['ticks']} ticks, cues at steps {s['cueSteps']}, "
            f"falls {s['falls']}, phases {s['phases']} -> {s['file']}", flush=True)


if __name__ == "__main__":
  main()
