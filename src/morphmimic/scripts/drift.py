"""Drift diagnostic of a checkpoint on a subset: every clip once from
its first frame, deterministic policy, eval env (no DR, no noise); per clip the xy and
heading error of the robot anchor against the reference anchor over time.

Example:
  uv run python -m morphmimic.scripts.drift --ckpt logs/run/ckpt_40000.pt \
    --subset-yaml $DATA/seed_g1_npz50/subsets/drift_long.yaml --gpu 0

Writes the per-clip summary csv to --out (default `<ckpt dir>/drift_<subset>_<iteration>.csv`)
and the per-step series to the same path with `.npz` (`<name>/xy`, `<name>/yaw`,
`<name>/dz`).
--interface human plays a `kp_interface` checkpoint through the human skeleton goal, as
in `morphmimic.scripts.eval`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from mjlab.utils.torch import configure_torch_backends

from morphmimic.data.manifest import load_subset
from morphmimic.eval.benchmark import make_eval_env
from morphmimic.eval.drift import XY_THRESHOLD, YAW_THRESHOLD_DEG, run_drift
from morphmimic.eval.policy_loader import InterfaceArgs, load_policy
from morphmimic.scripts.gpu_guard import select_gpu


@dataclass
class DriftArgs(InterfaceArgs):
  ckpt: str
  """Training checkpoint (`ckpt_<it>.pt` or `latest.pt`)."""
  subset_yaml: str
  """Subset whose clips are played."""
  gpu: int = 0
  """Physical GPU index (exposed to the process as cuda:0)."""
  allow_any_gpu: bool = False
  """Ignored (every GPU is allowed); kept so old command lines still parse."""
  num_envs: int = 64
  """Parallel envs (capped at the number of clips)."""
  out: str | None = None
  """Per-clip summary csv; default `<ckpt dir>/drift_<subset>_<iteration>[_<mode>].csv`."""
  library_dtype: str = "float32"
  robot: str | None = None
  """Robot to play on (default: the checkpoint's robot)."""


def format_summary(df: pd.DataFrame) -> str:
  n = len(df)
  lines = [
    f"clips                     {n} ({int(df['failed'].sum())} failed)",
    f"mean xy error             {df['mean_xy'].mean():.3f} m (median of clip means {df['mean_xy'].median():.3f})",
    f"final xy error            {df['final_xy'].mean():.3f} m",
    f"mean |heading| error      {df['mean_abs_yaw_deg'].mean():.1f} deg",
    f"final heading error       {df['final_yaw_deg'].abs().mean():.1f} deg (abs)",
    f"min height error          {df['min_dz'].mean():.3f} m (mean of clip minima; lowest clip {df['min_dz'].min():.3f})",
    f"clips over {XY_THRESHOLD} m         {int(df['t_xy_over'].notna().sum())}/{n}, median time {df['t_xy_over'].median():.1f} s",
    f"clips over {YAW_THRESHOLD_DEG:.0f} deg         {int(df['t_yaw_over'].notna().sum())}/{n}, median time {df['t_yaw_over'].median():.1f} s",
  ]
  return "\n".join(lines)


def main() -> None:
  args = tyro.cli(DriftArgs)
  select_gpu(args.gpu, args.allow_any_gpu)
  configure_torch_backends()
  policy_fn, meta = load_policy(
    args.ckpt,
    "cuda:0",
    args.interface,
    args.human_points,
    args.kp_horizon,
    goal_anchor=args.goal_anchor,
    human_heading=args.human_heading,
    human_drift=args.human_drift,
  )
  robot = args.robot or meta.robot
  num_clips = len(load_subset(args.subset_yaml))
  env = make_eval_env(
    args.subset_yaml,
    min(args.num_envs, num_clips),
    "cuda:0",
    library_dtype=args.library_dtype,
    robot=robot,
    human_root=args.human_root,
    goal_anchor=args.goal_anchor,
  )
  df, series = run_drift(env, policy_fn, verbose=True)
  suffix = "" if meta.mode == "motion" else f"_{meta.mode}"
  out = (
    Path(args.out)
    if args.out
    else Path(args.ckpt).parent
    / f"drift_{Path(args.subset_yaml).stem}_{meta.iteration}{suffix}.csv"
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out, index=False)
  np.savez_compressed(
    out.with_suffix(".npz"),
    **{f"{name}/xy": xy for name, (xy, _, _) in series.items()},
    **{f"{name}/yaw": yaw for name, (_, yaw, _) in series.items()},
    **{f"{name}/dz": dz for name, (_, _, dz) in series.items()},
  )
  print(f"[morphmimic] drift of {args.ckpt} (iteration {meta.iteration}, mode {meta.mode}) on {args.subset_yaml}")
  print(format_summary(df))
  print(f"[morphmimic] per-clip summary: {out}; series: {out.with_suffix('.npz')}")


if __name__ == "__main__":
  main()
