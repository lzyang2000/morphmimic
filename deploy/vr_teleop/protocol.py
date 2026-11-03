"""The streamer -> sim2sim policy node wire format: one msgpack `RefMessage` per ZMQ PUB
frame.

A message carries the controller mode, the calibration id, the device time of the last
sample, the stream rate, the target's name and a chunk of 50 Hz frames. Its `kind` field
names what a frame is (the retired five-point stream had none):
- "ref" (the GMR target, `gmr_streamer.GmrTarget`, the target named by the robot): one
  full reference state (`[240]`, the clip layout of `deploy/sim2sim`: canonical joints
  and velocities, the tracked bodies' poses (wxyz) and velocities, in the streamer's room
  frame);
- "human" (`human_target.HumanTarget`, named "human"): the operator on the uniform
  skeleton, the 24 SMPL slots' world poses `[24 * 7]` (position, then wxyz), the layout
  of `deploy/sim2sim/human.py`.
Arrays travel as little-endian float64 bytes so a roundtrip is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import msgpack
import numpy as np


def _pack_array(a: np.ndarray) -> bytes:
  return np.ascontiguousarray(a, dtype="<f8").tobytes()


def _unpack_array(b: bytes, shape: tuple[int, ...]) -> np.ndarray:
  return np.frombuffer(b, dtype="<f8").reshape(shape).copy()  # Copy: writable.


REF_VERSION = 1
# Frame width per message kind: a robot reference state, a human frame.
FRAME_DIM = {"ref": 240, "human": 24 * 7}


@dataclass
class RefFrame:
  """One 50 Hz frame of the streamer: a robot reference state or a human frame."""

  index: int
  state: np.ndarray  # [FRAME_DIM[kind]]


@dataclass
class RefMessage:
  mode: str
  calib_id: int
  t_ns: int
  fps: float
  robot: str
  frames: list[RefFrame]
  kind: str = "ref"


def encode_ref(msg: RefMessage) -> bytes:
  return msgpack.packb(
    {
      "kind": msg.kind,
      "v": REF_VERSION,
      "mode": msg.mode,
      "calib_id": msg.calib_id,
      "t_ns": msg.t_ns,
      "fps": msg.fps,
      "robot": msg.robot,
      "frames": [{"index": f.index, "state": _pack_array(f.state)} for f in msg.frames],
    }
  )


def decode_ref(b: bytes) -> RefMessage:
  d = msgpack.unpackb(b)
  if d.get("kind") not in FRAME_DIM or d.get("v") != REF_VERSION:
    raise ValueError(f"not a reference message (kind {d.get('kind')!r}, v {d.get('v')!r}),"
                     f" expected kind 'ref' or 'human' v {REF_VERSION}")
  width = FRAME_DIM[d["kind"]]
  return RefMessage(
    mode=d["mode"],
    calib_id=d["calib_id"],
    t_ns=d["t_ns"],
    fps=d["fps"],
    robot=d["robot"],
    frames=[RefFrame(f["index"], _unpack_array(f["state"], (width,))) for f in d["frames"]],
    kind=d["kind"],
  )
