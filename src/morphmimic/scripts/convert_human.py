"""Convert the SOMA BVHs of a robot npz set to 50 Hz human sidecars.

Example:
  uv run python -m morphmimic.scripts.convert_human --npz-root $DATA/seed_g1_npz50 \
    --bvh-root $DATA/seed/soma_uniform/bvh --workers 120

Each clip `<npz_root>/<date>/<stem>.npz` gets `<human_root>/<date>/<stem>.npz` from
`<bvh_root>/<date>/<stem>.bvh`; `human_root` defaults to `<npz_root>/human50`, where
`MotionLibrary` looks.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import tyro
from tqdm import tqdm

from morphmimic.data.human_convert import bvh_path, convert_human_clip, sidecar_path
from morphmimic.data.manifest import load_subset


def _convert_job(job: tuple[str, str, str, bool]) -> tuple[str, str]:
  """Pool worker: (clip, bvh, sidecar, overwrite) -> (clip, "converted" | "skipped" |
  "error: ...")."""
  clip, bvh, out, overwrite = job
  if not overwrite and os.path.exists(out):
    return clip, "skipped"
  try:
    convert_human_clip(bvh, out)
  # Recorded and printed at the end: one bad file must not stop a 142k-clip run.
  except Exception as e:  # noqa: BLE001
    return clip, f"error: {e!r}"
  return clip, "converted"


def main(
  npz_root: str,
  bvh_root: str,
  human_root: str | None = None,
  subset_yaml: str | None = None,
  workers: int = os.cpu_count() or 1,
  overwrite: bool = False,
) -> None:
  """Writes the human sidecar of every clip of `subset_yaml`, else of every
  `<npz_root>/<date>/<stem>.npz`.

  Args:
    npz_root: robot npz root (one sub-directory per date).
    bvh_root: SOMA Uniform BVH root (one sub-directory per date).
    human_root: sidecar root; default `<clip root>/human50` (`<npz_root>/human50`
      without a subset).
    subset_yaml: convert only this subset's clips.
    workers: number of conversion processes.
    overwrite: reconvert clips whose sidecar already exists.
  """
  if subset_yaml is not None:
    clips = load_subset(subset_yaml)
  else:
    clips = [str(p) for p in sorted(Path(npz_root).glob("*/*.npz"))]
  if not clips:
    raise FileNotFoundError(f"no <date>/<stem>.npz clips under {npz_root}")
  # `sidecar_path`'s default (`<clip root>/human50`) is where `MotionLibrary` looks.
  jobs = [
    (c, str(bvh_path(c, bvh_root)), str(sidecar_path(c, human_root)), overwrite)
    for c in clips
  ]
  with multiprocessing.Pool(workers) as pool:
    results = pool.imap_unordered(_convert_job, jobs, chunksize=8)
    status = dict(tqdm(results, total=len(jobs), desc="convert human"))
  failed = {c: s for c, s in sorted(status.items()) if s.startswith("error")}
  converted = sum(s == "converted" for s in status.values())
  skipped = sum(s == "skipped" for s in status.values())
  for clip, error in failed.items():
    print(f"[FAILED] {clip}: {error}")
  print(f"converted {converted}, skipped {skipped}, failed {len(failed)}")
  print(f"sidecars: {sidecar_path(clips[0], human_root).parent.parent}")


if __name__ == "__main__":
  tyro.cli(main)
