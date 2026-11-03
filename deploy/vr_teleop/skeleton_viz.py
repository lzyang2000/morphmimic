"""What the headset reports about the waist: PICO's 24-joint skeleton, no GMR, no physics.

  uv run --group teleop python -m deploy.vr_teleop.skeleton_viz [--replay session.npz]
    [--log angles.csv] [--no-viewer] [--print-every 0.5] [--fast]

Live from the headset (the XRoboToolkit SDK), or `--replay` a `streamer --record`
file (no SDK needed; `--fast` plays it as fast as it goes). A MuJoCo viewer shows the
skeleton PICO solves from the five measured devices (headset, two controllers, two
ankle trackers): joints as spheres, the five keypoint joints larger, bones, and a
frame triad (x red, y green, z blue, in the robot's x-forward / z-up convention) on the
Pelvis, Spine3 and Head. The terminal and `--log` carry, per sample, three relative
rotations as yaw / pitch / roll in degrees:

  waist           Spine3 relative to Pelvis: what a waist-orientation keypoint would
                  send;
  head            Head relative to Pelvis: the gaze, for telling the two apart;
  pelvis_vs_feet  the pelvis heading relative to the mean ankle heading: nothing on our
                  rig measures the pelvis, so this shows whether PICO turns its inferred
                  pelvis with the shoulders.

Protocol, hips fixed: twist the shoulders left and right with the controllers along;
head-only turns with the shoulders still; forward bend; side lean; squat. The waist
trace should follow the shoulders, not the head, and pelvis_vs_feet should stay put.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.vr_teleop.gmr_streamer import PICO_JOINTS, SMPL_PARENT, Body, pico_to_gmr
from deploy.vr_teleop.pico_source import PicoSource, ReplaySource

# GMR's `xrobot_to_g1` rotation offset of the Pelvis and Spine3 rows (wxyz): a PICO joint
# frame times this is the robot's x-forward, y-left, z-up body frame.
SMPL_TO_ROBOT = R.from_quat([-0.5, 0.5, -0.5, -0.5], scalar_first=True)
KEYPOINT_JOINTS = ("Pelvis", "Left_Ankle", "Right_Ankle", "Left_Wrist", "Right_Wrist")
TRIAD_JOINTS = ("Pelvis", "Spine3", "Head")
LOG_COLUMNS = (
  "t_s", "waist_yaw", "waist_pitch", "waist_roll", "head_yaw", "head_pitch",
  "head_roll", "pelvis_vs_feet",
)
JOINT_RADIUS, KEYPOINT_RADIUS, BONE_RADIUS = 0.02, 0.035, 0.008
TRIAD_LENGTH, TRIAD_RADIUS = 0.15, 0.005
_SKELETON = (0.75, 0.75, 0.75, 1.0)
_KEYPOINT = (1.0, 0.6, 0.1, 1.0)
_SPINE = (0.2, 0.5, 1.0, 1.0)
_AXES = ((1.0, 0.1, 0.1, 1.0), (0.1, 0.8, 0.1, 1.0), (0.2, 0.3, 1.0, 1.0))


def robot_rotation(quat_wxyz: np.ndarray) -> R:
  """A PICO joint's orientation as the robot-convention frame (x forward, z up)."""
  return R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True) * SMPL_TO_ROBOT


def relative_rotation(body: Body, joint: str, base: str = "Pelvis") -> R:
  """`joint`'s frame relative to `base`'s, in the robot convention."""
  return robot_rotation(body[base][1]).inv() * robot_rotation(body[joint][1])


def yaw_pitch_roll_deg(rot: R) -> tuple[float, float, float]:
  """Intrinsic Z-Y-X angles (yaw about up, pitch about left, roll about forward)."""
  y, p, r = rot.as_euler("ZYX", degrees=True)
  return float(y), float(p), float(r)


Angles = dict[str, tuple[float, float, float] | float]


def angles(body: Body) -> Angles:
  """`waist`, `head` (yaw, pitch, roll) and `pelvis_vs_feet` (yaw) of a body, degrees."""
  feet = [yaw_pitch_roll_deg(relative_rotation(body, f"{s}_Ankle"))[0] for s in ("Left", "Right")]
  pelvis_vs_feet = -float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(feet))))))
  return {
    "waist": yaw_pitch_roll_deg(relative_rotation(body, "Spine3")),
    "head": yaw_pitch_roll_deg(relative_rotation(body, "Head")),
    "pelvis_vs_feet": pelvis_vs_feet,
  }


class Geom(NamedTuple):
  kind: str  # "sphere" | "capsule"
  a: np.ndarray
  b: np.ndarray | None
  radius: float
  rgba: tuple[float, float, float, float]
  triad: tuple[str, int] | None = None  # (joint, axis) of a frame-triad capsule


def skeleton_geoms(body: Body) -> list[Geom]:
  """The skeleton as viewer geoms: a sphere per joint (keypoint joints larger and
  orange, Spine3 blue), a bone to each joint's SMPL parent, and the Pelvis, Spine3 and
  Head frame triads in the robot convention."""
  geoms = []
  for i, name in enumerate(PICO_JOINTS):
    pos = np.asarray(body[name][0], dtype=float)
    if name in KEYPOINT_JOINTS:
      geoms.append(Geom("sphere", pos, None, KEYPOINT_RADIUS, _KEYPOINT))
    else:
      geoms.append(Geom("sphere", pos, None, JOINT_RADIUS, _SPINE if name == "Spine3" else _SKELETON))
    if SMPL_PARENT[i] >= 0:
      parent = np.asarray(body[PICO_JOINTS[SMPL_PARENT[i]]][0], dtype=float)
      geoms.append(Geom("capsule", parent, pos, BONE_RADIUS, _SKELETON))
  for name in TRIAD_JOINTS:
    pos = np.asarray(body[name][0], dtype=float)
    axes = robot_rotation(body[name][1]).as_matrix()
    for k in range(3):
      geoms.append(Geom("capsule", pos, pos + TRIAD_LENGTH * axes[:, k], TRIAD_RADIUS,
                        _AXES[k], triad=(name, k)))
  return geoms


class AngleLog:
  """CSV of `LOG_COLUMNS`, one row per sample."""

  def __init__(self, path: Path) -> None:
    self._file = open(path, "w", newline="")
    self._writer = csv.writer(self._file)
    self._writer.writerow(LOG_COLUMNS)

  def add(self, t_s: float, a: Angles) -> None:
    values = (*a["waist"], *a["head"], a["pelvis_vs_feet"])
    self._writer.writerow([f"{t_s:.6f}"] + [f"{v:.3f}" for v in values])

  def close(self) -> None:
    self._file.close()


def _format(t_s: float, a: Angles) -> str:
  w, h, pf = a["waist"], a["head"], a["pelvis_vs_feet"]
  return (f"t={t_s:7.2f}s  waist y/p/r {w[0]:+6.1f} {w[1]:+6.1f} {w[2]:+6.1f}  |"
          f"  head {h[0]:+6.1f} {h[1]:+6.1f} {h[2]:+6.1f}  |  pelvis-feet {pf:+6.1f}")


_SCENE_XML = """
<mujoco>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <asset><texture type="2d" name="grid" builtin="checker" rgb1="0.85 0.85 0.85"
    rgb2="0.7 0.7 0.7" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/></asset>
  <worldbody><light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="4 4 0.05" material="grid"/></worldbody>
</mujoco>
"""


class _Viewer:
  """MuJoCo's passive viewer on an empty scene; `draw` replaces the user geoms."""

  def __init__(self) -> None:
    import mujoco
    import mujoco.viewer

    self._mujoco = mujoco
    self.model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    self.data = mujoco.MjData(self.model)
    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    self.viewer.cam.distance = 3.0
    self.viewer.cam.elevation = -15.0
    self.viewer.cam.azimuth = 150.0

  @property
  def running(self) -> bool:
    return self.viewer.is_running()

  def draw(self, body: Body) -> None:
    mujoco, scn = self._mujoco, self.viewer.user_scn
    scn.ngeom = 0
    for g in skeleton_geoms(body):
      if scn.ngeom >= scn.maxgeom:
        break
      geom = scn.geoms[scn.ngeom]
      scn.ngeom += 1
      rgba = np.asarray(g.rgba, np.float32)
      if g.kind == "sphere":
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([g.radius, 0, 0]),
                            g.a, np.eye(3).ravel(), rgba)
      else:
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, g.radius, g.a, g.b)
    self.viewer.cam.lookat[:] = np.asarray(body["Pelvis"][0], dtype=float)
    self.viewer.sync()

  def close(self) -> None:
    self.viewer.close()


def run(source, viewer: bool, log: Path | None, print_every_s: float = 0.5,
        until_finished: bool = False) -> int:
  """Reads `source` (a `PicoSource` or `ReplaySource`) until the viewer closes, or the
  recording ends with `until_finished`; prints the angles every `print_every_s` of
  sample time (never if negative), logs every sample to `log`. Returns the number of
  samples processed."""
  source.start()
  writer = AngleLog(log) if log is not None else None
  view = _Viewer() if viewer else None
  n, last_t, t0, last_print = 0, None, None, -np.inf
  try:
    while True:
      s = source.latest()
      if s is None or s.t_ns == last_t:
        if until_finished and getattr(source, "finished", False):
          break
        if view is not None and not view.running:
          break
        time.sleep(0.002)
        continue
      last_t = s.t_ns
      t0 = s.t_ns if t0 is None else t0
      body = pico_to_gmr(s.body)
      a = angles(body)
      t_s = (s.t_ns - t0) / 1e9
      if writer is not None:
        writer.add(t_s, a)
      if print_every_s >= 0.0 and t_s - last_print >= print_every_s:
        print(_format(t_s, a), flush=True)
        last_print = t_s
      if view is not None:
        view.draw(body)
      n += 1
  finally:
    source.stop()
    if writer is not None:
      writer.close()
    if view is not None:
      view.close()
  return n


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--replay", type=Path, default=None, help="a streamer --record file")
  parser.add_argument("--fast", action="store_true", help="replay as fast as it goes")
  parser.add_argument("--log", type=Path, default=None, help="CSV of the angles per sample")
  parser.add_argument("--no-viewer", action="store_true")
  parser.add_argument("--print-every", type=float, default=0.5, metavar="S")
  args = parser.parse_args()
  if args.replay is not None:
    source = ReplaySource(args.replay, realtime=not args.fast)
  else:
    source = PicoSource()
  n = run(source, viewer=not args.no_viewer, log=args.log, print_every_s=args.print_every,
          until_finished=args.replay is not None)
  print(f"{n} samples", file=sys.stderr)


if __name__ == "__main__":
  main()
