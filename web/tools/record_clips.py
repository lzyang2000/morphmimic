"""Record the sim2sim's robot trajectory (full `qpos`, once per control step).

  uv run --group export python web/tools/record_clips.py --deploy <deploy_dir> \\
    --assets <assets_dir> --clips <clip> ... --robots g1 x2 t2 \\
    --interfaces motion --out <out_dir>

Runs the unchanged launcher `deploy.run` on a plain clip (no `--from-stand`, so the sim
starts in the clip's first pose and `sim_node.run` plays it) and wraps `Tracker.record`,
which sees the sim after reset (frame 0) and after every control step. Writes
`<out>/<clip>/<robot>_<interface>.npy` (float64 [frames, nq]) and `.json` (the sim2sim
summary plus `nq`, `frames`). `<deploy_dir>` is an `export_onnx.py` folder and
`<assets_dir>` an `export_assets.py` folder. A robot the export names `<body>_<variant>`
(the released policy's g1_amber) is recorded under the page's name (`gen_live_assets.POLICY_ROBOT`).
Runs are strictly sequential: parallel runs collide on the ZMQ ports.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # Run as a script: `deploy` lives at the repo root.
  sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from deploy import run as sim_run  # noqa: E402
from deploy.sim2sim import sim_node  # noqa: E402
from web.tools.gen_live_assets import POLICY_ROBOT  # noqa: E402

DROPPED_KEYS = ("states", "ghost_qpos", "init_qpos")


def record_one(deploy: Path, assets: Path, robot: str, clip: str,
               interface: str) -> tuple[np.ndarray, dict]:
  """One closed-loop run: (qpos [frames, nq], the sim2sim summary with nq and frames).
  `frames` is the number of rows, which is the clip length unless the robot fell."""
  rows: list[np.ndarray] = []
  original = sim_node.Tracker.record

  def record(self, frame: int) -> bool:
    rows.append(self.sim.data.qpos.copy())
    return original(self, frame)

  argv = sys.argv
  sim_node.Tracker.record = record
  try:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "summary.json"
      sys.argv = ["run", "--deploy", str(deploy), "--assets", str(assets), "--robot", robot,
                  "--clip", clip, "--interface", interface, "--summary", str(path)]
      sim_run.main()
      summary = json.loads(path.read_text())
  finally:
    sim_node.Tracker.record = original
    sys.argv = argv
  qpos = np.asarray(rows, dtype=np.float64)
  for key in DROPPED_KEYS:
    summary.pop(key, None)
  summary["nq"] = int(qpos.shape[1])
  summary["frames"] = int(qpos.shape[0])
  return qpos, summary


def write_recording(out: Path, clip: str, robot: str, interface: str, qpos: np.ndarray,
                    summary: dict) -> None:
  folder = out / clip
  folder.mkdir(parents=True, exist_ok=True)
  np.save(folder / f"{robot}_{interface}.npy", qpos)
  (folder / f"{robot}_{interface}.json").write_text(json.dumps(summary, indent=1) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--deploy", type=Path, required=True)
  parser.add_argument("--assets", type=Path, required=True)
  parser.add_argument("--clips", nargs="+", required=True)
  parser.add_argument("--robots", nargs="+", default=["g1", "x2", "t2"])
  parser.add_argument("--interfaces", nargs="+", default=["motion"], choices=["motion", "human"])
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  for clip in args.clips:
    for robot in args.robots:
      for interface in args.interfaces:
        qpos, summary = record_one(args.deploy, args.assets, robot, clip, interface)
        page = {v: k for k, v in POLICY_ROBOT.items()}.get(robot, robot)
        write_recording(args.out, clip, page, interface, qpos, summary)
        print(f"[record] {clip} {robot} {interface}: {summary['result']}, "
              f"{summary['frames']} frames", flush=True)


if __name__ == "__main__":
  main()
