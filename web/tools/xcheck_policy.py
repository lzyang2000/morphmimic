"""Reference outputs of the released policy for tools/xcheck_policy.mjs.

For each robot, 64 random input sets drawn around that robot's own normaliser (mean plus a
fraction of the std, so the normalised inputs are O(1), as in running). Runs the full export
(.cache/live/policy/policy.onnx, the human inputs it does not read for action_motion at zero)
and writes the inputs and action_motion from Python onnxruntime to
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
  # The released export, not the page's pruned copy: the JS side runs public/live/policy.onnx, so
  # the comparison also checks that the pruning kept action_motion as it was.
  session = ort.InferenceSession(str(WEB / ".cache/live/policy/policy.onnx"),
                                 providers=["CPUExecutionProvider"])
  types = {"tensor(float)": np.float32, "tensor(bool)": bool}
  unread = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], types[i.type])
            for i in session.get_inputs()}  # the human inputs action_motion does not read
  rng = np.random.default_rng(SEED)
  doc = {"cases": CASES, "robots": []}
  for spec in robots:
    inp = {k: np.asarray(v, np.float32) for k, v in spec["inputs"].items()}
    def around(mean, var, shape):
      std = np.sqrt(var)
      return (mean + 0.7 * std * rng.standard_normal(shape)).astype(np.float32)
    obs = around(np.tile(inp["obsMean"], 10), np.tile(inp["obsVar"], 10), (CASES, 930))
    gm = around(inp["goalMean"], inp["goalVar"], (CASES, 670))
    feeds_static = {"obs_mean": inp["obsMean"], "obs_var": inp["obsVar"],
                    "goal_mean": inp["goalMean"], "goal_var": inp["goalVar"],
                    "morph": inp["morph"]}
    motion = []
    for i in range(CASES):
      feeds = {**unread, "obs_history": obs[i:i + 1], "goal_motion": gm[i:i + 1], **feeds_static}
      motion.append(session.run(["action_motion"], feeds)[0][0])
    f = lambda a: [[float(f"{x:.9g}") for x in row] for row in a]
    doc["robots"].append({
      "name": spec["name"],
      "obs_history": f(obs), "goal_motion": f(gm), "action_motion": f(motion),
    })
    print(spec["name"], "motion action std:", float(np.std(motion)))
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "reference.json").write_text(json.dumps(doc))
  print("wrote", OUT / "reference.json")


if __name__ == "__main__":
  main()
