"""Human command traces for `scripts/render.py --kp-trace`.

Per enabled slot of the point mask (`HUMAN_JOINTS` order for the human skeleton): a
sphere at the current target, a bright line through the preview frames the policy
sees, and a fading trail of where the target has been. The point count is the mask's
length. Positions are world-frame numpy arrays; drawing goes through mjlab's
`DebugVisualizer` (`add_sphere`, `add_cylinder`), so this module has no simulator
dependency.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from morphmimic.constants import (
  GOAL_K,
  HUMAN_FIVE_IDX,
  HUMAN_HEAD,
  HUMAN_SPINE3,
  NUM_HUMAN,
)

# The five-point slots (pelvis, left ankle, right ankle, left wrist, right wrist), the
# chest and the head get their own colours; the other slots cycle a muted palette.
_SLOT_COLORS: dict[int, tuple[float, float, float]] = {
  **dict(
    zip(
      HUMAN_FIVE_IDX,
      (
        (1.0, 0.85, 0.1),
        (1.0, 0.45, 0.1),
        (0.1, 0.8, 0.9),
        (0.9, 0.15, 0.25),
        (0.2, 0.4, 1.0),
      ),
    )
  ),
  HUMAN_SPINE3: (0.6, 0.9, 0.3),
  HUMAN_HEAD: (0.95, 0.95, 0.95),
}
_OTHER_COLORS = ((0.55, 0.55, 0.6), (0.7, 0.6, 0.45), (0.45, 0.6, 0.7))
KP_TRACE_COLORS: tuple[tuple[float, float, float], ...] = tuple(
  _SLOT_COLORS.get(p, _OTHER_COLORS[p % len(_OTHER_COLORS)]) for p in range(NUM_HUMAN)
)
"""RGB per slot of the human skeleton (`HUMAN_JOINTS` order)."""
TARGET_RADIUS = 0.04
PREVIEW_RADIUS = 0.016
TRAIL_RADIUS = 0.010
PREVIEW_ALPHA = 0.95
TRAIL_ALPHA = (0.3, 0.9)  # oldest -> newest segment


@dataclass(frozen=True)
class Segment:
  point: int
  start: np.ndarray
  end: np.ndarray
  kind: Literal["trail", "preview"]
  alpha: float


def trace_segments(
  trail: np.ndarray, future: np.ndarray, preview: int, point_mask: Sequence[bool]
) -> list[Segment]:
  """Line segments of the command over P = `len(point_mask)` points. `trail` [T, P, 3]
  holds the past targets, oldest first, with the current target last; `future`
  [K, P, 3] the preview frames, of which the first `preview` (1..K) are drawn, chained
  from the current target. Only the points with `point_mask` on get segments."""
  if not 1 <= preview <= GOAL_K:
    raise ValueError(f"Expected 1 <= preview <= {GOAL_K}, got {preview}.")
  n = len(point_mask)
  trail = np.asarray(trail, dtype=np.float64)
  future = np.asarray(future, dtype=np.float64)
  if trail.ndim != 3 or trail.shape[1:] != (n, 3) or trail.shape[0] < 1:
    raise ValueError(f"Expected trail [T>=1, {n}, 3], got {trail.shape}.")
  if future.ndim != 3 or future.shape[1:] != (n, 3) or future.shape[0] < preview:
    raise ValueError(f"Expected future [K>={preview}, {n}, 3], got {future.shape}.")
  segs: list[Segment] = []
  n_trail = trail.shape[0] - 1
  alphas = np.linspace(TRAIL_ALPHA[0], TRAIL_ALPHA[1], n_trail) if n_trail > 1 else [TRAIL_ALPHA[1]]
  for p in range(n):
    if not point_mask[p]:
      continue
    for i in range(n_trail):
      segs.append(Segment(p, trail[i, p], trail[i + 1, p], "trail", float(alphas[i])))
    prev = trail[-1, p]
    for k in range(preview):
      segs.append(Segment(p, prev, future[k, p], "preview", PREVIEW_ALPHA))
      prev = future[k, p]
  return segs


class KeypointTrace:
  """Draws the command of the `point_mask` slots once per rendered frame (one point
  per mask entry, coloured by `KP_TRACE_COLORS`), keeping the last `trail_len` targets
  as the trail. Make a new one per clip."""

  def __init__(self, trail_len: int, preview: int, point_mask: Sequence[bool]) -> None:
    if not 1 <= len(point_mask) <= len(KP_TRACE_COLORS):
      raise ValueError(
        f"Expected 1..{len(KP_TRACE_COLORS)} point-mask entries, got {len(point_mask)}."
      )
    self._trail: deque[np.ndarray] = deque(maxlen=max(1, trail_len))
    self._preview = preview
    self._mask = tuple(bool(m) for m in point_mask)

  def draw(self, visualizer, current: np.ndarray, future: np.ndarray) -> None:
    """`current` [P, 3]: the target of this frame; `future` [K, P, 3]: the preview
    (P = the mask's length)."""
    current = np.asarray(current, dtype=np.float64)
    n = len(self._mask)
    if current.shape != (n, 3):
      raise ValueError(f"Expected current [{n}, 3], got {current.shape}.")
    self._trail.append(current)
    trail = np.stack(list(self._trail))
    for s in trace_segments(trail, future, self._preview, self._mask):
      rgb = KP_TRACE_COLORS[s.point]
      radius = PREVIEW_RADIUS if s.kind == "preview" else TRAIL_RADIUS
      visualizer.add_cylinder(s.start, s.end, radius, (*rgb, s.alpha))
    for p in range(n):
      if self._mask[p]:
        visualizer.add_sphere(current[p], TARGET_RADIUS, (*KP_TRACE_COLORS[p], 1.0))
