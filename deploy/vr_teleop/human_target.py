"""PICO body -> the operator on the uniform skeleton, for the policy node's human interface
(`streamer.py --target human`): no GMR, no robot.

The human goal was trained on the SOMA Uniform skeleton only (one body for every clip,
1.77 m), so the operator's motion is moved onto it: each frame, PICO's 24 joint
orientations are put on the uniform skeleton's bones (`human_skeleton.json`) and its
forward kinematics gives the positions. Whatever the operator's height and proportions,
the policy sees the uniform body doing the operator's joint motion.

Frames. PICO's `get_body_joints_pose` gives the 24 SMPL joints (`sim2sim.human.PICO_JOINTS`,
the goal's slot order) in Unity's y-up frame, xyz and xyzw, with SMPL's rest convention
(identity at the T-pose) but in its own axes: x right, y up, z back, where the training
sidecars' SMPL frames have x left, y up, z forward. So a joint's orientation in ours is
`A q Ry(pi)`, its position `A p`, with `A = Rz(pi) Rx(pi/2)` (y-up to z-up, then facing
the other way, which only turns the room). Measured on a real session (10-24, 46 s):
with this one constant the standing operator's joint rotations relative to their
parents are within 2-15 degrees of the SEED stand frame's (the shoulders 35, holding the
controllers), and GMR's hand-tuned PICO-to-G1 offsets imply the same joint frames within
1-8 degrees (pelvis, chest, arms). No per-joint offset is applied: one fitted to PICO's
own bone directions would tilt the chest by 19 degrees.

Calibration (A+B+X+Y, the operator standing): the scale is the uniform stand's pelvis
height above its ankles over the operator's; the root follows PICO's pelvis times the
scale, and a height offset puts the skeleton's ankles at the stand's ankle height, as
the GMR path does (the lower ankle at the stand's lower one). The offset follows the
floor (`floor.FloorTracker`: the lowest ankle in stance), so a slope, stairs or PICO's
height drift do not sink the goal's feet. The policy node aligns the heading, as for
GMR.

  python -m deploy.vr_teleop.human_target --human-root $DATA/seed_g1_npz50/human50
writes `human_skeleton.json` from the training sidecars (the bones and the stand).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deploy.sim2sim.human import (
  HUMAN_PELVIS,
  SKELETON_FILE,
  SMPL_PARENT,
  human_heading,
  join_frames,
  split_frames,
  turn_human,
)
from deploy.sim2sim.obs import quat_apply, quat_inv, quat_mul
from deploy.vr_teleop.floor import FloorTracker
from deploy.vr_teleop.protocol import RefFrame, RefMessage, encode_ref

# `A`: Unity's y-up to z-up (+90 degrees about x), then 180 degrees about z (wxyz).
_A = quat_mul(np.array([0.0, 0.0, 0.0, 1.0]), np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]))
_RY_PI = np.array([0.0, 0.0, 1.0, 0.0])  # PICO's joint axes to SMPL's: 180 degrees about y.
_ANKLES = [7, 8]
# Calibration needs the operator standing: the pelvis at least this far above the ankles
# (m; the uniform body's is 0.94).
MIN_PELVIS_HEIGHT = 0.5
# The SEED frame the robots' neutral stand (`gmr_configs/stand.json`) is retargeted from.
STAND_SOURCE = ("230413/dance_hiphop_scooby_doo_R_002__A320", 36)


def pico_to_human(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """PICO's [24, 7] (Unity xyz, xyzw) -> positions [24, 3] (m, z-up, PICO's size) and
  orientations [24, 4] (wxyz, SMPL's rest convention as the sidecars, w >= 0)."""
  quat = quat_mul(_A, quat_mul(body[:, [6, 3, 4, 5]], _RY_PI))
  return quat_apply(_A, body[:, :3]), quat * np.where(quat[:, :1] < 0, -1.0, 1.0)


def human_to_pico(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
  """The inverse of `pico_to_human`, for synthetic sessions: [24, 3], [24, 4] -> [24, 7]."""
  a_inv = quat_inv(_A)
  q = quat_mul(a_inv, quat_mul(quat, quat_inv(_RY_PI)))
  return np.concatenate([quat_apply(a_inv, pos), q[:, [1, 2, 3, 0]]], -1)


def forward_kinematics(root: np.ndarray, quat: np.ndarray, bones: np.ndarray) -> np.ndarray:
  """Slot positions [24, 3] of the skeleton with bones `bones` [24, 3] (each slot's
  offset from its SMPL parent in the parent's frame), the slot orientations `quat`
  [24, 4] and the pelvis at `root` [3]. SMPL's parents precede their children."""
  pos = np.zeros((len(SMPL_PARENT), 3))
  pos[0] = root
  for c in range(1, len(SMPL_PARENT)):
    p = SMPL_PARENT[c]
    pos[c] = pos[p] + quat_apply(quat[p], bones[c])
  return pos


class HumanTarget:
  """The streamer target sending the operator on the uniform skeleton (`RefMessage`
  kind "human", frames [168])."""

  name = "human"

  def __init__(self, skeleton_file: Path = SKELETON_FILE) -> None:
    doc = json.loads(Path(skeleton_file).read_text())
    self.bones = np.asarray(doc["bones"])
    self.pelvis_height = float(doc["pelvis_height"])
    # The floor is the lower ankle (`FloorTracker`), so it goes to the stand's lower ankle.
    self.ankle_z = float(np.min(np.asarray(doc["stand"]["pos"])[_ANKLES, 2]))
    self.scale = 1.0
    self.z_offset = 0.0
    self.floor = FloorTracker()
    self.calibrated = False

  def calibrate(self, body: np.ndarray) -> None:
    """Fix the scale (the uniform pelvis height above the ankles over the operator's)
    and the height offset (the skeleton's ankles at the stand's height)."""
    pos, quat = pico_to_human(body)
    height = pos[HUMAN_PELVIS, 2] - pos[_ANKLES, 2].mean()
    if height < MIN_PELVIS_HEIGHT:  # Not standing: the scale would be wrong, even negative.
      print(f"[streamer] human: calibration skipped, the pelvis is {height:.2f} m above the"
            " ankles; stand straight and press the combo again", flush=True)
      return
    self.scale = self.pelvis_height / height
    self.floor.reset(self._skeleton(pos, quat)[_ANKLES])
    self.z_offset = self.ankle_z - self.floor.z
    self.calibrated = True
    print(f"[streamer] human: calibrated, scale {self.scale:.3f} (the operator's pelvis "
          f"{self.pelvis_height / self.scale:.3f} m above the ankles), height offset "
          f"{self.z_offset:+.3f} m", flush=True)

  def frame(self, index: int, body: np.ndarray) -> RefFrame:
    pos, quat = pico_to_human(body)
    return RefFrame(index, join_frames(self._pos(pos, quat), quat))

  def message(self, mode: str, calib_id: int, t_ns: int, fps: float,
              frames: list) -> tuple[RefMessage, bytes]:
    msg = RefMessage(mode=mode, calib_id=calib_id, t_ns=t_ns, fps=fps, robot=self.name,
                     frames=frames, kind="human")
    return msg, encode_ref(msg)

  def _pos(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """The skeleton with its ankles on the followed floor at the stand's height."""
    skeleton = self._skeleton(pos, quat)
    self.z_offset = self.ankle_z - self.floor.update(skeleton[_ANKLES])
    return skeleton + np.array([0.0, 0.0, self.z_offset])

  def _skeleton(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return forward_kinematics(self.scale * pos[HUMAN_PELVIS], quat, self.bones)


def skeleton_doc(human_root: Path) -> dict:
  """`human_skeleton.json` from the training sidecars under `human_root`: the uniform
  skeleton's bones (the median over the stand clip of each slot's offset from its
  parent in the parent's frame; the uniform body is rigid to 1 mm) and the neutral stand
  (`STAND_SOURCE`, its pelvis moved to x = y = 0 and turned to face +x)."""
  stem, frame = STAND_SOURCE
  d = np.load(Path(human_root) / f"{stem}.npz")
  pos, quat = d["human_pos_w"].astype(np.float64), d["human_quat_w"].astype(np.float64)
  parent = np.array([max(p, 0) for p in SMPL_PARENT])
  bones = np.median(quat_apply(quat_inv(quat[:, parent]), pos - pos[:, parent]), axis=0)
  bones[0] = 0.0
  stand = join_frames(pos[frame], quat[frame])
  stand = turn_human(stand, -human_heading(stand), pos[frame, HUMAN_PELVIS, :2])
  stand_pos, stand_quat = split_frames(stand)
  stand_quat = stand_quat * np.where(stand_quat[:, :1] < 0, -1.0, 1.0)  # w >= 0, as the sidecars
  ankle_z = float(stand_pos[_ANKLES, 2].mean())
  return {
    "source": f"{stem}.npz frame {frame} (human50 sidecar)",
    "bones": np.round(bones, 6).tolist(),
    "pelvis_height": round(float(stand_pos[HUMAN_PELVIS, 2]) - ankle_z, 6),
    "ankle_z": round(ankle_z, 6),
    "stand": {"pos": np.round(stand_pos, 6).tolist(), "quat": np.round(stand_quat, 6).tolist()},
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="write human_skeleton.json")
  parser.add_argument("--human-root", type=Path, required=True,
                      help="the training sidecars (<root>/<date>/<stem>.npz)")
  parser.add_argument("--out", type=Path, default=SKELETON_FILE)
  args = parser.parse_args()
  args.out.write_text(json.dumps(skeleton_doc(args.human_root.expanduser()), indent=1) + "\n")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
