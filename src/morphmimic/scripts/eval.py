"""Benchmark a checkpoint on a subset (paper App. F): every clip once from
its first frame, deterministic policy, no DR or noise, floor friction 1.0.

Example:
  uv run python -m morphmimic.scripts.eval --ckpt logs/smoke_tiny/latest.pt \
    --subset-yaml $DATA/seed_g1_npz50/subsets/tiny.yaml --num-envs 256

Prints the summary (overall, by manifest category, and the drift block of
`morphmimic.scripts.drift`) and writes the per-clip table to --out (default
`<ckpt dir>/eval_<subset>_<iteration>.csv`); its last columns are the per-clip drift
columns of `scripts.drift` (`mean_xy` .. `t_yaw_over`). Pick the GPU
with --gpu.

--interface human evaluates a checkpoint of a `kp_interface` run through the human
skeleton goal with the fixed mask --human-points
{full,five,six,three} and --kp-horizon h (visible preview frames, 1..10);
--no-human-heading zeroes the relative orientation (the human chest's orientation, or
the pelvis's with five points, relative to the robot torso). The clips need human sidecars (--human-root, default
`<clip root>/human50`). The default file is then
`eval_<subset>_<iteration>_human-<points>-h<h>.csv`, `-noyaw` added before `.csv`
without the heading. The default, --interface motion, is the motion benchmark (any
checkpoint; the human flags are ignored).

Transfer: --robot runs the checkpoint on another body, through the
translator, with that body's PD gains; --policy-joint-frame keeps the checkpoint
robot's action scale and default pose, --builtin-pd swaps DC motors for implicit PD,
--morph-robot feeds the actor another robot's morphology vector; --morph-mix a:b:alpha
a point between two robots' vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import tyro
import yaml
from mjlab.utils.torch import configure_torch_backends

from morphmimic.data.manifest import load_manifest
from morphmimic.eval.benchmark import evaluate, summarize
from morphmimic.eval.drift import XY_THRESHOLD, YAW_THRESHOLD_DEG
from morphmimic.eval.policy_loader import InterfaceArgs
from morphmimic.scripts.gpu_guard import select_gpu


@dataclass
class EvalArgs(InterfaceArgs):
  ckpt: str
  """Training checkpoint (`ckpt_<it>.pt` or `latest.pt`)."""
  subset_yaml: str
  """Subset whose clips are evaluated."""
  gpu: int = 0
  """Physical GPU index (exposed to the process as cuda:0)."""
  allow_any_gpu: bool = False
  """Ignored (every GPU is allowed); kept so old command lines still parse."""
  num_envs: int = 4096
  """Parallel envs (capped at the number of clips)."""
  out: str | None = None
  """Per-clip csv; default `<ckpt dir>/eval_<subset>_<iteration>.csv`, with
  `_human-<points>-h<horizon>` (and `-noyaw` without the heading) appended before
  `.csv` for the human interface."""
  manifest: str | None = None
  """Manifest with clip categories; default `<subset root>/manifest.csv`."""
  library_dtype: str = "float32"
  """Storage dtype of the reference motions (float16 halves memory; bfloat16 loses
  cm-level precision at world scale)."""
  robot: str | None = None
  """Robot to evaluate on (default: the checkpoint's robot)."""
  builtin_pd: bool = False
  """Built-in implicit PD on every joint (the X2's DC motors replaced, same gains)."""
  policy_joint_frame: bool = False
  """With --robot another body: the canonical joints keep the checkpoint robot's
  action scale and default pose (each action is the same angle on both bodies);
  default: the evaluated robot's own, as its deploy json has them."""
  morph_robot: str | None = None
  """Checkpoint with the morphology input: act with this robot's actor morphology
  vector instead of the stored one (its own robot's)."""
  morph_mix: str | None = None
  """`a:b:alpha`: act with the vector (1 - alpha) * vec(a) + alpha * vec(b), a point
  between two training bodies. Not with --morph-robot."""
  morph_file: str | None = None
  """A `.pt` file with an `actor` tensor: act with that actor morphology vector (a
  fitted one). Not with --morph-robot or --morph-mix."""
  norm_from: str | None = None
  """Another checkpoint whose observation and goal normalisers replace this one's
  (e.g. the T2 rank of the same multi-robot run)."""
  imu_body: str | None = None
  """Read the policy's gyro and gravity on this tracked body instead of the robot's
  IMU body (sim-only check, e.g. waist_yaw_link on the T2)."""


def load_morph_file(path: str) -> torch.Tensor:
  """The `actor` tensor of a `.pt` file (a dict), float32 and flat; ValueError when the
  file has no `actor` entry."""
  data = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(data, dict) or "actor" not in data:
    raise ValueError(f"{path}: expected a dict with an 'actor' tensor")
  return torch.as_tensor(data["actor"], dtype=torch.float32).reshape(-1)


def parse_morph_mix(text: str) -> tuple[str, str, float]:
  """`"a:b:alpha"` -> `(a, b, alpha)`; ValueError on any other shape."""
  parts = text.split(":")
  if len(parts) != 3:
    raise ValueError(f"--morph-mix expects a:b:alpha, got {text!r}")
  return parts[0], parts[1], float(parts[2])


def default_out_path(
  ckpt: str, subset_yaml: str, iteration: int, mode: str = "motion"
) -> Path:
  """`<ckpt dir>/eval_<subset>_<iteration>.csv`, with `_<mode>` before `.csv` unless
  `mode` is "motion" (`PolicyMeta.mode`)."""
  suffix = "" if mode == "motion" else f"_{mode}"
  return Path(ckpt).parent / f"eval_{Path(subset_yaml).stem}_{iteration}{suffix}.csv"


def default_manifest_path(subset_yaml: str) -> Path:
  with open(subset_yaml) as f:
    return Path(yaml.safe_load(f)["root"]) / "manifest.csv"


def format_summary(summary: dict[str, Any]) -> str:
  """The `summarize` dict as an overall block, a per-category table and the drift
  block (as `scripts.drift` prints it)."""
  lines = [
    f"clips               {summary['num_clips']}",
    f"failures            {summary['failures']} ({summary['failure_rate']:.1%})",
    f"mean norm. return   {summary['mean_norm_return']:.4f}",
    f"median norm. return {summary['median_norm_return']:.4f}",
  ]
  table = pd.DataFrame(
    {
      "clips": summary["clips_by_category"],
      "failures": summary["failures_by_category"],
      "mean_norm_return": summary["mean_norm_return_by_category"],
    }
  )
  table["failure_rate"] = table["failures"] / table["clips"]
  table = table.sort_values("clips", ascending=False)
  table = table[["clips", "failures", "failure_rate", "mean_norm_return"]]
  table.index.name = "category"
  formatters = {"failure_rate": "{:.1%}".format, "mean_norm_return": "{:.4f}".format}
  lines += ["", table.to_string(formatters=formatters)]
  drift = summary["drift"]
  n = summary["num_clips"]
  lines += [
    "",
    f"mean xy error        {drift['mean_xy']:.3f} m",
    f"final xy error       {drift['final_xy']:.3f} m",
    f"mean |heading| error {drift['mean_abs_yaw_deg']:.1f} deg",
    f"clips over {XY_THRESHOLD} m    {drift['clips_xy_over']}/{n}",
    f"clips over {YAW_THRESHOLD_DEG:.0f} deg   {drift['clips_yaw_over']}/{n}",
  ]
  return "\n".join(lines)


def main() -> None:
  args = tyro.cli(EvalArgs)
  select_gpu(args.gpu, args.allow_any_gpu)
  configure_torch_backends()  # TF32 matmuls, as in training.
  df = evaluate(
    args.ckpt,
    args.subset_yaml,
    "cuda:0",
    num_envs=args.num_envs,
    library_dtype=args.library_dtype,
    verbose=True,
    robot=args.robot,
    interface=args.interface,
    human_points=args.human_points,
    kp_horizon=args.kp_horizon,
    human_root=args.human_root,
    human_heading=args.human_heading,
    human_drift=args.human_drift,
    goal_anchor=args.goal_anchor,
    builtin_pd=args.builtin_pd,
    policy_joint_frame=args.policy_joint_frame,
    morph_robot=args.morph_robot,
    morph_mix=parse_morph_mix(args.morph_mix) if args.morph_mix else None,
    morph_vector=load_morph_file(args.morph_file) if args.morph_file else None,
    norm_from=args.norm_from,
    imu_body=args.imu_body,
  )
  out = (
    Path(args.out)
    if args.out
    else default_out_path(
      args.ckpt, args.subset_yaml, df.attrs["iteration"], df.attrs["mode"]
    )
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out, index=False)

  manifest_path = (
    Path(args.manifest) if args.manifest else default_manifest_path(args.subset_yaml)
  )
  if manifest_path.is_file():
    manifest = load_manifest(str(manifest_path))
  else:
    print(f"[morphmimic] no manifest at {manifest_path}; categories are unknown")
    manifest = pd.DataFrame({"name": pd.Series(dtype=str), "category": []})
  summary = summarize(df, manifest)
  print(
    f"[morphmimic] benchmark of {args.ckpt} (iteration {df.attrs['iteration']}, "
    f"mode {df.attrs['mode']}) on {args.subset_yaml}"
  )
  print(format_summary(summary))
  print(f"[morphmimic] per-clip results: {out}")


if __name__ == "__main__":
  main()
