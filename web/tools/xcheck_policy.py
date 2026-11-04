"""Reference outputs of the released policy for tools/xcheck_policy.mjs.

For each robot, 64 random input sets drawn around that robot's own normaliser (mean plus a
fraction of the std, so the normalised inputs are O(1), as in running). Human inputs come with
random valid masks (the full or five-point slot setting, a prefix of the 10-frame window seen,
the heading block on or off). Runs public/live/policy.onnx (the export, copied) and writes the
inputs and both outputs from Python onnxruntime to .cache/xcheck_policy/reference.json. Run from web/ (`npm run xcheck-policy` does).
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

WEB = Path(__file__).resolve().parents[1]
OUT = WEB / ".cache" / "xcheck_policy"
CASES = 64
SEED = 20261023


def main() -> None:
  robots = json.loads((WEB / "public/live/robots.json").read_text())["robots"]
  session = ort.InferenceSession(str(WEB / "public/live/policy.onnx"),
                                 providers=["CPUExecutionProvider"])
  rng = np.random.default_rng(SEED)
  doc = {"cases": CASES, "robots": []}
  for spec in robots:
    inp = {k: np.asarray(v, np.float32) for k, v in spec["inputs"].items()}
    def around(mean, var, shape):
      std = np.sqrt(var)
      return (mean + 0.7 * std * rng.standard_normal(shape)).astype(np.float32)
    obs = around(np.tile(inp["obsMean"], 10), np.tile(inp["obsVar"], 10), (CASES, 930))
    gm = around(inp["goalMean"], inp["goalVar"], (CASES, 670))
    gh = around(inp["humanMean"], inp["humanVar"], (CASES, 2160))
    hh = around(inp["headingMean"], inp["headingVar"], (CASES, 60))
    point = np.zeros((CASES, 24), bool)
    point[: CASES // 2] = True  # full
    point[CASES // 2:, [0, 7, 8, 20, 21]] = True  # five
    frame = np.arange(10)[None] < rng.integers(1, 11, CASES)[:, None]
    hmask = rng.random(CASES) < 0.8
    feeds_static = {"obs_mean": inp["obsMean"], "obs_var": inp["obsVar"],
                    "goal_mean": inp["goalMean"], "goal_var": inp["goalVar"],
                    "human_mean": inp["humanMean"], "human_var": inp["humanVar"],
                    "heading_mean": inp["headingMean"], "heading_var": inp["headingVar"],
                    "morph": inp["morph"]}
    motion, human = [], []
    for i in range(CASES):
      feeds = {"obs_history": obs[i:i + 1], "goal_motion": gm[i:i + 1],
               "goal_human": gh[i:i + 1], "human_point_mask": point[i:i + 1],
               "human_frame_mask": frame[i:i + 1], "human_heading": hh[i:i + 1],
               "human_heading_mask": hmask[i:i + 1], **feeds_static}
      m, h = session.run(["action_motion", "action_human"], feeds)
      motion.append(m[0])
      human.append(h[0])
    f = lambda a: [[float(f"{x:.9g}") for x in row] for row in a]
    doc["robots"].append({
      "name": spec["name"],
      "obs_history": f(obs), "goal_motion": f(gm), "goal_human": f(gh), "human_heading": f(hh),
      "human_point_mask": point.astype(int).tolist(), "human_frame_mask": frame.astype(int).tolist(),
      "human_heading_mask": hmask.astype(int).tolist(),
      "action_motion": f(motion), "action_human": f(human),
    })
    print(spec["name"], "motion / human action std:", float(np.std(motion)), float(np.std(human)))
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "reference.json").write_text(json.dumps(doc))
  print("wrote", OUT / "reference.json")


if __name__ == "__main__":
  main()
