"""BVH reader: parses the hierarchy and runs forward kinematics of the channels.

Copied from `deploy/vr_teleop/fit_gmr_configs.py::load_bvh` (the training package does
not import from `deploy`), plus the world orientations it computes along the way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _parse_hierarchy(
  header: str,
) -> tuple[list[str | None], list[int], list[np.ndarray], list[list[str]]]:
  """Joint names (None for end sites), parent indices (-1 at a root), OFFSETs and
  CHANNELS of a BVH HIERARCHY block, in file order."""
  names: list[str | None] = []
  parents: list[int] = []
  offsets: list[np.ndarray] = []
  channels: list[list[str]] = []
  stack: list[int] = []
  for line in header.splitlines():
    tok = line.split()
    if not tok:
      continue
    if tok[0] in ("ROOT", "JOINT", "End"):
      names.append(tok[1] if tok[0] != "End" else None)
      parents.append(stack[-1] if stack else -1)
      offsets.append(np.zeros(3))
      channels.append([])
    elif tok[0] == "{":
      stack.append(len(names) - 1)
    elif tok[0] == "}":
      stack.pop()
    elif tok[0] == "OFFSET":
      offsets[-1] = np.array([float(x) for x in tok[1:4]])
    elif tok[0] == "CHANNELS":
      channels[-1] = tok[2:]
  return names, parents, offsets, channels


def load_bvh_offsets(path: Path) -> dict[str, np.ndarray]:
  """Joint name -> its hierarchy OFFSET [3] (BVH units) of a BVH."""
  header = Path(path).read_text().split("MOTION")[0]
  names, _, offsets, _ = _parse_hierarchy(header)
  return {n: o for n, o in zip(names, offsets) if n}


def load_bvh_poses(
  path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], float]:
  """World joint positions [T, J, 3] (BVH units), world orientations [T, J, 4] wxyz,
  joint name -> index, and the frame time (s) of a BVH.

  End sites take an index (they have no name). Rotation channels are applied in the
  file's channel order as intrinsic Euler angles in degrees; a position channel
  replaces the matching offset component.
  """
  header, motion = Path(path).read_text().split("MOTION")
  names, parents, offsets, channels = _parse_hierarchy(header)
  lines = motion.strip().splitlines()
  frame_time = float(lines[1].split()[-1])
  values = np.array([[float(x) for x in line.split()] for line in lines[2:]])
  num = len(values)
  pos = np.zeros((num, len(names), 3))
  quat = np.zeros((num, len(names), 4))
  rots: list[Rotation] = []
  col = 0
  for j, chans in enumerate(channels):
    local = np.tile(offsets[j], (num, 1))
    rot = Rotation.identity(num)
    if chans:
      v = values[:, col : col + len(chans)]
      col += len(chans)
      for k, ch in enumerate(chans):
        if ch.endswith("position"):
          local[:, "XYZ".index(ch[0])] = v[:, k]
      axes = [(k, ch[0]) for k, ch in enumerate(chans) if ch.endswith("rotation")]
      if axes:
        rot = Rotation.from_euler(
          "".join(a for _, a in axes), v[:, [k for k, _ in axes]], degrees=True
        )
    if parents[j] < 0:
      pos[:, j], world = local, rot
    else:
      pos[:, j] = pos[:, parents[j]] + rots[parents[j]].apply(local)
      world = rots[parents[j]] * rot
    rots.append(world)
    quat[:, j] = np.roll(world.as_quat(), 1, axis=-1)
  return pos, quat, {n: i for i, n in enumerate(names) if n}, frame_time


def write_bvh(
  path: Path,
  names: list[str | None],
  parents: list[int],
  offsets: list[np.ndarray],
  channels: list[list[str]],
  frames: np.ndarray,
  frame_time: float,
) -> None:
  """Writes a BVH in the layout `_parse_hierarchy` reads: joints in file order (None
  names are end sites; parents come before children), their OFFSETs and CHANNELS, then
  `frames` [T, sum of channels] (degrees for rotations) at `frame_time` s, as `%.6f`."""
  lines = ["HIERARCHY"]
  depth = [0] * len(names)
  open_depths: list[int] = []
  for j, name in enumerate(names):
    depth[j] = depth[parents[j]] + 1 if parents[j] >= 0 else 0
    while open_depths and open_depths[-1] >= depth[j]:
      lines.append(" " * open_depths.pop() + "}")
    pad = " " * depth[j]
    if parents[j] < 0:
      lines.append(f"{pad}ROOT {name}")
    else:
      lines.append(f"{pad}JOINT {name}" if name else f"{pad}End Site")
    lines.append(pad + "{")
    lines.append(f"{pad} OFFSET " + " ".join(f"{v:.6f}" for v in offsets[j]))
    if channels[j]:
      lines.append(f"{pad} CHANNELS {len(channels[j])} " + " ".join(channels[j]))
    open_depths.append(depth[j])
  while open_depths:
    lines.append(" " * open_depths.pop() + "}")
  lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {frame_time:.6f}"]
  lines += [" ".join(f"{v:.6f}" for v in row) for row in np.asarray(frames)]
  Path(path).write_text("\n".join(lines) + "\n")
