"""Check the human sidecars of a subset against its robot clips.

Example:
  uv run python -m morphmimic.scripts.check_human --subset-yaml subsets/1h.yaml \
    --out ~/tmp/check_human.csv

Prints the clips without a sidecar and those whose frame counts differ, and counts the
clips whose feet's lowest z lies outside [-0.05, 0.10] m (clips with a sidecar); exits
1 on any missing sidecar or frame mismatch.
"""

from __future__ import annotations

import sys

import pandas as pd
import tyro
from tqdm import tqdm

from morphmimic.data.human_convert import check_clip
from morphmimic.data.manifest import load_subset

FEET_Z_RANGE = (-0.05, 0.10)


def main(subset_yaml: str, human_root: str | None = None, out: str | None = None) -> None:
  """Runs `check_clip` on every clip of `subset_yaml`.

  Args:
    subset_yaml: subset of robot clips.
    human_root: sidecar root; default `<clip root>/human50`.
    out: write the per-clip table to this csv.
  """
  clips = load_subset(subset_yaml)
  df = pd.DataFrame(
    [check_clip(c, human_root) for c in tqdm(clips, desc="check human")]
  )
  missing = df[~df["sidecar"]]
  found = df[df["sidecar"]]
  mismatches = found[~found["frames_match"]]
  lo, hi = FEET_Z_RANGE
  feet_off = found[~found["feet_min_z"].between(lo, hi)]
  for _, row in missing.iterrows():
    print(f"[MISSING] {row['name']}: no sidecar")
  for _, row in mismatches.iterrows():
    print(f"[MISMATCH] {row['name']}: robot {row['robot_frames']} frames,"
          f" human {row['human_frames']}")
  print(f"{len(df)} clips, {len(missing)} missing sidecars, {len(mismatches)} mismatches,"
        f" {len(feet_off)} feet_min_z outside [{lo}, {hi}] m")
  print(f"height ratio (robot / human pelvis): median {df['height_ratio'].median():.3f},"
        f" min {df['height_ratio'].min():.3f}, max {df['height_ratio'].max():.3f}")
  print(f"pelvis xy rms at that scale (m): median {df['pelvis_xy_rms_m'].median():.3f},"
        f" max {df['pelvis_xy_rms_m'].max():.3f}")
  if out is not None:
    df.to_csv(out, index=False)
    print(f"table: {out}")
  if len(missing) or len(mismatches):
    sys.exit(1)


if __name__ == "__main__":
  tyro.cli(main)
