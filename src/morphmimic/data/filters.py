"""Clip filters of App. E: name keywords and kinematic feasibility.

Both filters return a short reason string when a clip is rejected and None otherwise.
"""

from __future__ import annotations

import re

import numpy as np

# Clip-name keywords for motions that need environment support or props the simulator
# does not have. Matched case-insensitively as whole words on the clip name with "_"
# and "-" read as spaces, so "box" rejects "come_down_50cm_box_R_001" but not
# "shadow_boxing", and "sit" does not match "transit". The first matching keyword in
# this order is the reason.
NAME_FILTER_KEYWORDS: tuple[str, ...] = (
  # Paper App. E / SONIC keyword list.
  "stair",
  "stairs",
  "ladder",
  "box",
  "chair",
  "sit",
  "sitting",
  "climb",
  "climbing",
  "jump off",
  "look over fence",
  "bench",
  "table",
  "step up",
  "platform",
  "crawl under",
  # Equivalent terms found in SEED clip names: elevated surfaces, leaning on walls,
  # crutches, riding, seats.
  "50cm",
  "on the edge",
  "ledge",
  "leaning",
  "crutch",
  "crutches",
  "horse riding",
  "seat",
)

# Sessions whose clip names carry no prop or terrain information (LAFAN1 names are
# `<action><n>_subject<k>_<clip>`): only the feasibility filter applies.
NAMED_OK_SESSIONS: tuple[str, ...] = ("lafan",)

_NAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
  (kw, re.compile(r"\b" + r"\s+".join(re.escape(w) for w in kw.split()) + r"\b"))
  for kw in NAME_FILTER_KEYWORDS
)

# Feasibility thresholds (App. E, G1 retarget at 50 Hz). z_min is the lowest body-link
# origin height and foot_dist the signed foot-to-floor distance, both in metres. The
# *_Z and FLOATING_*_DIST heights are for the G1 and scale with a robot's height_scale.
ELEVATED_Z = 0.20
ELEVATED_SECONDS = 0.6
PENETRATION_Z = -0.12
PENETRATION_SUSTAINED_Z = -0.08
PENETRATION_FRACTION = 0.8
FLOATING_FOOT_DIST = 0.02
FLOATING_SECONDS = 0.5
FLOATING_UP_Z = 0.9
FLOATING_PELVIS_Z = 0.6
FLOATING_SPEED = 0.3


def name_filter_reject(name: str, session: str | None = None) -> str | None:
  """First keyword of NAME_FILTER_KEYWORDS found in the clip name, else None. Clips
  of a NAMED_OK_SESSIONS session (`<session>/<clip>`) always pass."""
  if session in NAMED_OK_SESSIONS:
    return None
  text = re.sub(r"[_\-]+", " ", name.lower())
  for kw, pattern in _NAME_PATTERNS:
    if pattern.search(text):
      return kw
  return None


def _longest_run(mask: np.ndarray) -> int:
  """Length of the longest run of consecutive True values."""
  best = run = 0
  for value in mask:
    run = run + 1 if value else 0
    best = max(best, run)
  return best


def feasibility_reject(
  z_min: np.ndarray,
  foot_dist: np.ndarray,
  pelvis_z: np.ndarray,
  pelvis_up_z: np.ndarray,
  body_speed: np.ndarray,
  fps: int = 50,
  height_scale: float = 1.0,
) -> str | None:
  """App. E feasibility rules on per-frame clip statistics.

  Args:
    z_min: [T] lowest body-link origin height.
    foot_dist: [T, 2] left/right foot distance to the floor.
    pelvis_z: [T] pelvis height.
    pelvis_up_z: [T] world z component of the pelvis z axis (1 when upright).
    body_speed: [T] mean linear speed over bodies.
    fps: frame rate of the arrays.
    height_scale: robot leg length relative to the G1's; multiplies the height
      thresholds (ELEVATED_Z, PENETRATION_Z, PENETRATION_SUSTAINED_Z,
      FLOATING_FOOT_DIST, FLOATING_PELVIS_Z).

  Returns:
    "elevated_support", "ground_penetration", "floating_feet" or None.
  """

  def min_frames(seconds: float) -> int:
    return int(np.ceil(seconds * fps - 1e-6))

  s = height_scale
  if _longest_run(z_min > ELEVATED_Z * s) >= min_frames(ELEVATED_SECONDS):
    return "elevated_support"
  if np.any(z_min < PENETRATION_Z * s) or (
    np.mean(z_min < PENETRATION_SUSTAINED_Z * s) > PENETRATION_FRACTION
  ):
    return "ground_penetration"
  floating = (
    np.all(foot_dist > FLOATING_FOOT_DIST * s, axis=1)
    & (pelvis_up_z > FLOATING_UP_Z)
    & (pelvis_z > FLOATING_PELVIS_Z * s)
    & (body_speed < FLOATING_SPEED)
  )
  if _longest_run(floating) >= min_frames(FLOATING_SECONDS):
    return "floating_feet"
  return None
