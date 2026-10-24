"""Reference outputs of public/live/policy.onnx for tools/xcheck_policy.mjs.

For each robot, 64 random input sets drawn around that robot's own normaliser (mean plus a
fraction of the std, so the normalised inputs are O(1), as in running), with valid random
masks (at least one point on; the frame mask a prefix of the 10-frame window, as kp_horizon
makes it). Writes the inputs and both outputs from Python onnxruntime to
.cache/xcheck_policy/reference.json. Run from web/ (`npm run xcheck-policy` does).
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
    gk = around(inp["kpMean"], inp["kpVar"], (CASES, 450))
    point = rng.random((CASES, 5)) < 0.7
    point[np.arange(CASES), rng.integers(0, 5, CASES)] = True
    frame = np.arange(10)[None] < rng.integers(1, 11, CASES)[:, None]
    feeds_static = {"obs_mean": inp["obsMean"], "obs_var": inp["obsVar"],
                    "goal_mean": inp["goalMean"], "goal_var": inp["goalVar"],
                    "kp_mean": inp["kpMean"], "kp_var": inp["kpVar"], "morph": inp["morph"]}
    motion, keypoint = [], []
    for i in range(CASES):
      feeds = {"obs_history": obs[i:i + 1], "goal_motion": gm[i:i + 1],
               "goal_keypoint": gk[i:i + 1], "kp_point_mask": point[i:i + 1],
               "kp_frame_mask": frame[i:i + 1], **feeds_static}
      m, k = session.run(["action_motion", "action_keypoint"], feeds)
      motion.append(m[0])
      keypoint.append(k[0])
    f = lambda a: [[float(f"{x:.9g}") for x in row] for row in a]
    doc["robots"].append({
      "name": spec["name"],
      "obs_history": f(obs), "goal_motion": f(gm), "goal_keypoint": f(gk),
      "kp_point_mask": point.astype(int).tolist(), "kp_frame_mask": frame.astype(int).tolist(),
      "action_motion": f(motion), "action_keypoint": f(keypoint),
    })
    print(spec["name"], "motion/keypoint action std:",
          float(np.std(motion)), float(np.std(keypoint)))
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "reference.json").write_text(json.dumps(doc))
  print("wrote", OUT / "reference.json")


if __name__ == "__main__":
  main()
