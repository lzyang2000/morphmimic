"""The operator's floor height, followed through the session.

Calibration puts the operator's ankles at the stand's height once; the goal heights are
absolute after that, so walking down a slope or stairs (or PICO's height drift) sinks
the goal's feet below the flat floor the policy was trained on, and the robot crouches
to lower its pelvis. `FloorTracker` follows the floor instead: the lowest ankle that has
been still for a while (in stance), low-passed. While neither ankle is (a jump, a run's
flight phase) the floor holds, so jumps keep their height; a squat keeps the feet still
at the floor, so it stays a squat.
"""

from __future__ import annotations

import numpy as np

from deploy.vr_teleop.constants import FPS

# An ankle slower than this (m/s) is in stance. PICO's ankle jitter at 50 Hz is up to
# ~0.15 m/s standing still; a swing foot moves at 1-4 m/s.
STILL_SPEED = 0.3
# Low-pass time constant (s): a step down (a stair, the next foot on a slope) reaches
# the goal over ~3 TAU (after STANCE_FRAMES); walking down a 10-degree slope at 1 m/s,
# the floor ends each stance ~2 cm above the foot.
TAU = 0.2
_ALPHA = 1.0 - np.exp(-1.0 / (FPS * TAU))
# Frames an ankle must stay still to count as in stance: a jump's apex is still for
# ~0.06 s, a walking stance lasts ~0.6 s.
STANCE_FRAMES = round(0.15 * FPS)


class FloorTracker:
  """The floor height under ankles streamed at `FPS`."""

  def __init__(self) -> None:
    self.z = 0.0
    self._prev: np.ndarray | None = None
    self._still = np.zeros(2, int)  # Frames each ankle has been still.

  def reset(self, ankles: np.ndarray) -> None:
    """The floor at the calibration stand: the lower of the ankles [2, 3]."""
    self.z = float(ankles[:, 2].min())
    self._prev = ankles.copy()
    self._still[:] = STANCE_FRAMES

  def update(self, ankles: np.ndarray) -> float:
    """Take one frame's ankles [2, 3] and return the floor height."""
    still = np.linalg.norm(ankles - self._prev, axis=1) * FPS < STILL_SPEED
    self._prev = ankles.copy()
    self._still = np.where(still, self._still + 1, 0)
    stance = self._still >= STANCE_FRAMES
    if stance.any():
      self.z += _ALPHA * (float(ankles[stance, 2].min()) - self.z)
    return self.z
