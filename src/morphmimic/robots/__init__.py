"""Robot registry: `get_robot(name)` -> RobotSpec for the G1, X2 Ultra and T2
(`ROBOTS`, the training robots) and for the variants (`VARIANTS`: `g1_amber`, the G1
with the waist roll/pitch damping at 3.0).

Specs are built on first use; importing this package compiles no model.
"""

from __future__ import annotations

from functools import cache

from morphmimic.robots.spec import RobotSpec, Translator

ROBOTS: tuple[str, ...] = ("g1", "x2", "t2")
"""The training robots: the morphology scaling is fixed over these."""
VARIANTS: tuple[str, ...] = ("g1_amber",)
"""Bodies outside the morphology scaling's training set; scaled against `ROBOTS`."""

__all__ = [
  "ROBOTS",
  "VARIANTS",
  "RobotSpec",
  "Translator",
  "check_imu_body",
  "get_robot",
]


@cache
def get_robot(name: str) -> RobotSpec:
  if name == "g1":
    from morphmimic.robots.g1 import g1_spec

    return g1_spec()
  if name == "x2":
    from morphmimic.robots.x2 import x2_spec

    return x2_spec()
  if name == "t2":
    from morphmimic.robots.t2 import t2_spec

    return t2_spec()
  if name == "g1_amber":
    from morphmimic.robots.g1_amber import g1_amber_spec

    return g1_amber_spec()
  raise ValueError(f"unknown robot {name!r}; expected one of {ROBOTS + VARIANTS}")


def check_imu_body(ckpt: dict, path: object) -> None:
  """Raises ValueError when checkpoint `ckpt` (at `path`) was trained with its robot's
  gyro and gravity read on another body than `RobotSpec.imu_body`. A checkpoint
  without the `imu_body` entry read them on tracked slot 0: the same body on the G1
  and X2, the T2's waist link instead of its trunk."""
  robot = ckpt.get("cfg", {}).get("robot", "g1")
  spec = get_robot(robot)
  saved = ckpt.get("imu_body", spec.tracked_bodies[0])
  if saved != spec.imu_body:
    raise ValueError(
      f"checkpoint {path} was trained with the {robot} IMU on {saved}; this code reads"
      f" it on {spec.imu_body}, so its policy inputs differ"
    )
