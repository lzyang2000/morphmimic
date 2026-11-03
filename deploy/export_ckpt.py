"""Inference-only copy of a training checkpoint: what `load_policy` reads, nothing else.

Keeps the policy (motion and, for a `kp_interface` run, the human encoder), its
normalisers (`human_norm` for the human interface), the morphology entry, the training
config and the iteration. Drops the world model, critics, optimiser state, reset
sampler and RNG (about 245 of 300 MB for a full-set three-robot run), so the result
loads in `eval`, `render`, `play` and `deploy.export_onnx` but cannot resume training.

  uv run python -m deploy.export_ckpt <ckpt_40000.pt> <out.pt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

KEEP = (
  "policy",
  "obs_norm",
  "goal_norm",
  "human_norm",
  "heading_norm",
  "morph",
  "cfg",
  "iteration",
  "goal_version",
  "imu_body",
  "kp_goal_version",
)


def export(src: str | Path, dst: str | Path) -> dict:
  """Writes the `KEEP` entries of checkpoint `src` to `dst`; returns them."""
  ckpt = torch.load(src, map_location="cpu")
  out = {k: ckpt[k] for k in KEEP if k in ckpt}
  Path(dst).parent.mkdir(parents=True, exist_ok=True)
  torch.save(out, dst)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("src")
  parser.add_argument("dst")
  args = parser.parse_args()
  out = export(args.src, args.dst)
  print(f"{args.dst}: {', '.join(out)} (iteration {out['iteration']}, robot {out['cfg'].get('robot')})")


if __name__ == "__main__":
  main()
