"""Pack the recorded sim2sim qpos trajectories for the browser.

  uv run --project .. --group export python tools/pack_recordings.py [--cache DIR] [--out DIR]

Reads <cache>/recordings/<clip>/<robot>_<interface>.{npy,json} and <cache>/policy.txt, writes
<out>/rec/index.json and one <out>/rec/<clip>.bin per clip: float32 little-endian qpos blocks
([frames, nq] row-major, one per available robot/interface) at the byte `offset`s in the index.

Every recording is aligned so the robot starts at frame 0 facing +x at world y = lateral
offset (X2 -1.5, G1 0, T2 +1.5: on screen X2 left, G1 middle, T2 right). Only the simulated qpos is shipped, never reference data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WEB = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = WEB / ".cache"
DEFAULT_OUT = WEB / "public"

FPS = 50
ROBOTS = ("g1", "x2", "t2")
INTERFACES = ("motion",)
LATERAL = {"g1": 0.0, "x2": -1.5, "t2": 1.5}

# (id, label) in sidebar order.
CLIPS = (
    ("walking_random_direction_R_001__A432", "Walk"),
    ("dance_basic_padeburee_270_R_loop_001__A324", "Ballet: pas de bourrée"),
    ("dance_basic_turn_v1_360_R_loop_001__A325", "Ballet: turns"),
    ("dance_hiphop_shuffle_sequence_v_001_R_003__A306", "Street: shuffle"),
    ("dance_hiphop_hip_hop_i_R_001__A313", "Street: hip hop"),
    ("dance_latino_jive_sequence_R_002__A313", "Jive"),
    ("dance_macarena_003__A467", "Macarena"),
    ("dance_western_ridding_alone_R_003__A306", "Country line dance"),
    ("dancing_routine_V002_002__A365", "Dance routine"),
    ("victory_dance_robot_dance_180_R_003__A306", "Robot dance"),
)


def align(qpos: np.ndarray, lateral: float) -> np.ndarray:
  """Rotate about z so frame 0's base yaw is 0 and put frame 0's base xy at (0, lateral).

  qpos[:, 0:3] is the base position and qpos[:, 3:7] the base quaternion (wxyz); the joint
  columns are untouched. Returns a new array.
  """
  q = np.array(qpos, dtype=np.float64, copy=True)
  w, x, y, z = q[0, 3:7]
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  c, s = np.cos(-yaw), np.sin(-yaw)
  rel = q[:, 0:2] - q[0, 0:2]
  q[:, 0] = c * rel[:, 0] - s * rel[:, 1]
  q[:, 1] = s * rel[:, 0] + c * rel[:, 1] + lateral
  # q' = qz(-yaw) * q
  hw, hz = np.cos(-yaw / 2), np.sin(-yaw / 2)
  bw, bx, by, bz = q[:, 3].copy(), q[:, 4].copy(), q[:, 5].copy(), q[:, 6].copy()
  q[:, 3] = hw * bw - hz * bz
  q[:, 4] = hw * bx - hz * by
  q[:, 5] = hw * by + hz * bx
  q[:, 6] = hw * bz + hz * bw
  q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
  return q


def _load(rec_dir: Path, robot: str, iface: str):
  """Return (qpos, jointErr, nq) for a completed recording, else (None, None, nq-or-None)."""
  stem = rec_dir / f"{robot}_{iface}"
  try:
    summary = json.loads(stem.with_suffix(".json").read_text())
  except FileNotFoundError:
    return None, None, None
  nq = summary.get("nq")
  if summary.get("result") != "completed" or not stem.with_suffix(".npy").exists():
    return None, None, nq
  qpos = np.load(stem.with_suffix(".npy"))
  if qpos.ndim != 2 or len(qpos) == 0 or (nq is not None and qpos.shape[1] != nq):
    return None, None, nq
  return qpos, float(summary["joint_err_rms_rad"]), int(qpos.shape[1])


def pack(cache: Path = DEFAULT_CACHE, out: Path = DEFAULT_OUT) -> dict:
  cache, out = Path(cache), Path(out)
  rec_out = out / "rec"
  rec_out.mkdir(parents=True, exist_ok=True)
  policy = (cache / "policy.txt").read_text().strip()

  clips = []
  for clip_id, label in CLIPS:
    rec_dir = cache / "recordings" / clip_id
    blob = bytearray()
    robots, longest = {}, 0
    for robot in ROBOTS:
      robots[robot] = {}
      for iface in INTERFACES:
        qpos, err, nq = _load(rec_dir, robot, iface)
        if qpos is None:
          robots[robot][iface] = {"offset": 0, "frames": 0, "available": False,
                                  "jointErr": None, "nq": nq}
          continue
        aligned = align(qpos, LATERAL[robot])
        robots[robot][iface] = {"offset": len(blob), "frames": int(len(aligned)),
                                "available": True, "jointErr": err, "nq": nq}
        blob += aligned.astype("<f4").tobytes()
        longest = max(longest, len(aligned))
    bin_path = rec_out / f"{clip_id}.bin"
    if blob:
      bin_path.write_bytes(bytes(blob))
    elif bin_path.exists():
      bin_path.unlink()
    clips.append({"id": clip_id, "label": label,
                  "duration": (longest - 1) / FPS if longest else 0.0, "robots": robots})

  index = {"policy": policy, "fps": FPS, "clips": clips}
  (rec_out / "index.json").write_text(json.dumps(index, indent=1))
  return index


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = ap.parse_args()
  index = pack(args.cache, args.out)
  n = sum(e["available"] for c in index["clips"] for r in c["robots"].values() for e in r.values())
  total = sum(1 for c in index["clips"] for r in c["robots"].values() for _ in r.values())
  print(f"packed {n}/{total} recordings for {len(index['clips'])} clips ({index['policy']})")
  for c in index["clips"]:
    for robot, ifaces in c["robots"].items():
      for iface, e in ifaces.items():
        if not e["available"]:
          print(f"  unavailable: {c['id']} {robot} {iface}")


if __name__ == "__main__":
  main()
