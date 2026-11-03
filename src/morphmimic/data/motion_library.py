"""Multi-clip reference-motion store, with the bins of the reset sampler.

All clips of a subset are concatenated along one global frame axis and kept on one
device. A frame index is global; `frame_clip` maps it to its clip, and reads never cross
a clip boundary (`future` holds the clip's last frame). Each clip is cut into bins of
`bin_size` frames for reset sampling; see `MotionLibrary` for the clip's last bin.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from morphmimic.constants import FPS, HUMAN_JOINTS, LOCAL_DIM, TRACKED_BODY_NAMES
from morphmimic.data.human_convert import sidecar_path
from morphmimic.data.manifest import load_subset
from morphmimic.model.state import (
  FrameMode,
  StateParts,
  canonicalize,
  join_state,
  state_frame,
)

if TYPE_CHECKING:
  from morphmimic.robots import RobotSpec

_JOINT_KEYS = ("joint_pos", "joint_vel")
_BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
_ROOT_KEYS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")
_HUMAN_KEYS = ("human_pos_w", "human_quat_w")
# Half-precision storage: the |x|, |y| of world body positions from which the spacing
# of representable values (ulp) reaches 1/64 m = 1.6 cm (float16: 16 m, 10-bit
# mantissa; bfloat16: 2 m, 7-bit mantissa). `MotionLibrary` warns beyond it.
_HALF_XY_LIMIT_M = {torch.float16: 16.0, torch.bfloat16: 2.0}


class MotionLibrary:
  """Reference frames of many npz clips (Task 3 schema) as concatenated tensors.

  Motion tensors are stored in `dtype`; `state` and `future` always return float32.
  float16 halves the memory of large subsets (the full set); with its 10-bit mantissa
  world xy positions within 16 m of the origin are stored on a grid finer than
  1.6 cm, and the library warns when a clip goes farther. bfloat16 reaches 1.6 cm
  steps at 2 m already, so it is not recommended. Body tensors hold `body_names` in
  that order; the `root_*` tensors hold the floating base. Every clip needs at least
  2 frames.

  Without `robot`, joints are read in npz order (the G1's) and the floating base is
  the body `body_names[0]`. With `robot`, the npz joints are matched by name to
  `robot.joint_names` and translated to the canonical 29 joints, the bodies are
  `robot.tracked_bodies`, and the floating base is the body `robot.root_body`.

  Bins tile each clip in order with `bin_size` frames. The remainder of a clip forms
  its last bin, unless it is shorter than `bin_size // 2` and the clip has a full bin
  before it: then it is merged into that previous bin. Bins thus hold between
  `bin_size // 2` and `bin_size + bin_size // 2 - 1` frames (25 to 74 at 1 s, 50 Hz),
  except that a clip shorter than `bin_size // 2` is a single short bin. A tiny last
  bin would only start very short episodes and dominate return-based sampling.

  Human poses (`human_pos_w`, `human_quat_w`, the 24 `HUMAN_JOINTS` slots) come from
  each clip's sidecar (`human_convert.sidecar_path`, under `human_root` when given).
  Every clip has one, or none does (both `None`); a partial set is an error.
  """

  joint_pos: Tensor  # [N, 29]
  joint_vel: Tensor  # [N, 29]
  body_pos_w: Tensor  # [N, B, 3]
  body_quat_w: Tensor  # [N, B, 4] wxyz
  body_lin_vel_w: Tensor  # [N, B, 3]
  body_ang_vel_w: Tensor  # [N, B, 3]
  root_pos_w: Tensor  # [N, 3]
  root_quat_w: Tensor  # [N, 4] wxyz
  root_lin_vel_w: Tensor  # [N, 3]
  root_ang_vel_w: Tensor  # [N, 3]
  human_pos_w: Tensor | None  # [N, 24, 3]
  human_quat_w: Tensor | None  # [N, 24, 4] wxyz

  def __init__(
    self,
    clip_paths: list[str],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    body_names: tuple[str, ...] = TRACKED_BODY_NAMES,
    bin_seconds: float = 1.0,
    robot: RobotSpec | None = None,
    human_root: str | None = None,
  ) -> None:
    if not clip_paths:
      raise ValueError("MotionLibrary needs at least one clip.")
    self.device = torch.device(device)
    self.body_names = tuple(body_names if robot is None else robot.tracked_bodies)
    root_body = self.body_names[0] if robot is None else robot.root_body
    self.fps = FPS
    self.clip_names = [Path(p).stem for p in clip_paths]

    sidecars = [sidecar_path(p, human_root) for p in clip_paths]
    no_sidecar = [str(p) for p in sidecars if not p.exists()]
    has_human = len(no_sidecar) < len(sidecars)
    if has_human and no_sidecar:
      raise ValueError(
        f"{len(no_sidecar)} of {len(sidecars)} clips have no human sidecar (all or"
        f" none must), e.g. {no_sidecar[:10]}"
      )
    parts: dict[str, list[np.ndarray]] = {
      k: [] for k in (*_JOINT_KEYS, *_BODY_KEYS, *_ROOT_KEYS, *_HUMAN_KEYS)
    }
    lengths = []
    for path, side in tqdm(
      zip(clip_paths, sidecars),
      total=len(clip_paths),
      desc="motion library",
      disable=len(clip_paths) <= 100,
    ):
      with np.load(path) as z:
        fps = int(z["fps"])
        if fps != FPS:
          raise ValueError(f"{path}: fps {fps}, expected {FPS}.")
        names = [str(n) for n in z["body_names"]]
        missing = [n for n in (*self.body_names, root_body) if n not in names]
        if missing:
          raise ValueError(f"{path}: missing bodies {missing}.")
        idx = [names.index(n) for n in self.body_names]
        root = names.index(root_body)
        joint_idx = slice(None)
        if robot is not None:
          joints = [str(n) for n in z["joint_names"]]
          if sorted(joints) != sorted(robot.joint_names):
            raise ValueError(
              f"{path}: joints do not match {robot.name}; missing"
              f" {sorted(set(robot.joint_names) - set(joints))}, unexpected"
              f" {sorted(set(joints) - set(robot.joint_names))}."
            )
          joint_idx = [joints.index(n) for n in robot.joint_names]
        frames = len(z["joint_pos"])
        if frames < 2:
          raise ValueError(
            f"{path}: {frames} frame(s); a clip needs at least 2 (one transition)."
          )
        for key in _JOINT_KEYS:
          parts[key].append(z[key][:, joint_idx])
        for key, root_key in zip(_BODY_KEYS, _ROOT_KEYS):
          body = z[key]
          parts[key].append(body[:, idx])
          parts[root_key].append(body[:, root])
        lengths.append(frames)
      if has_human:
        with np.load(side) as h:
          human_fps = int(h["fps"])
          if human_fps != FPS:
            raise ValueError(f"{path}: human sidecar fps {human_fps}, expected {FPS}")
          if tuple(str(n) for n in h["human_joint_names"]) != HUMAN_JOINTS:
            raise ValueError(f"{side}: human_joint_names are not HUMAN_JOINTS.")
          human_frames = len(h["human_pos_w"])
          if human_frames != frames:
            raise ValueError(
              f"{path}: human sidecar has {human_frames} frames, the clip {frames}"
            )
          for key in _HUMAN_KEYS:
            parts[key].append(h[key])

    if dtype in _HALF_XY_LIMIT_M:
      xy_max = max(float(np.abs(pos[..., :2]).max()) for pos in parts["body_pos_w"])
      if xy_max >= _HALF_XY_LIMIT_M[dtype]:
        warnings.warn(
          f"MotionLibrary: body positions reach |x|, |y| = {xy_max:.2f} m; {dtype}"
          f" storage rounds them to steps of >= 1.6 cm beyond"
          f" {_HALF_XY_LIMIT_M[dtype]:g} m (use float32, or float16 for clips"
          " within 16 m).",
          stacklevel=2,
        )

    def cat(key: str, to: torch.dtype = dtype) -> Tensor:
      return torch.as_tensor(np.concatenate(parts.pop(key))).to(self.device, to)

    if robot is None:
      self.joint_pos = cat("joint_pos")
      self.joint_vel = cat("joint_vel")
    else:
      # Native -> canonical once for all frames, in float32 on the target device.
      tr = robot.translator
      self.joint_pos = tr.to_canonical(cat("joint_pos", torch.float32)).to(dtype)
      self.joint_vel = tr.to_canonical_delta(cat("joint_vel", torch.float32)).to(dtype)
    self.body_pos_w = cat("body_pos_w")
    self.body_quat_w = cat("body_quat_w")
    self.body_lin_vel_w = cat("body_lin_vel_w")
    self.body_ang_vel_w = cat("body_ang_vel_w")
    self.root_pos_w = cat("root_pos_w")
    self.root_quat_w = cat("root_quat_w")
    self.root_lin_vel_w = cat("root_lin_vel_w")
    self.root_ang_vel_w = cat("root_ang_vel_w")
    self.human_pos_w = cat("human_pos_w") if has_human else None
    self.human_quat_w = cat("human_quat_w") if has_human else None

    # Clips.
    self.num_clips = len(lengths)
    self.num_frames = sum(lengths)
    clip_ids = torch.arange(self.num_clips, device=self.device)
    self.clip_len = torch.tensor(lengths, dtype=torch.long, device=self.device)
    self.clip_start = torch.cumsum(self.clip_len, 0) - self.clip_len
    self.clip_seconds = self.clip_len.float() / self.fps
    self.frame_clip = torch.repeat_interleave(
      clip_ids, self.clip_len, output_size=self.num_frames
    )

    # Bins (see the class docstring): full bins per clip, plus the remainder as its
    # own bin unless it is short and can be merged into a full bin before it.
    self.bin_size = round(bin_seconds * self.fps)
    full_bins = self.clip_len // self.bin_size
    rest = self.clip_len % self.bin_size
    own_bin = (rest >= self.bin_size // 2) | ((rest > 0) & (full_bins == 0))
    clip_bins = full_bins + own_bin.long()
    self.num_bins = int(clip_bins.sum())
    clip_first_bin = torch.cumsum(clip_bins, 0) - clip_bins
    self.bin_clip = torch.repeat_interleave(
      clip_ids, clip_bins, output_size=self.num_bins
    )
    bin_in_clip = (
      torch.arange(self.num_bins, device=self.device) - clip_first_bin[self.bin_clip]
    )
    self.bin_start = self.clip_start[self.bin_clip] + bin_in_clip * self.bin_size
    # A clip's last bin runs to the clip's end (merged remainder included).
    last = bin_in_clip == clip_bins[self.bin_clip] - 1
    clip_stop = (self.clip_start + self.clip_len)[self.bin_clip]
    self.bin_len = torch.where(
      last, clip_stop - self.bin_start, torch.full_like(self.bin_start, self.bin_size)
    )

  @classmethod
  def from_yaml(cls, path: str, **kw) -> MotionLibrary:
    """Library of the clips of a subset yaml (see `manifest.build_subset`)."""
    return cls(load_subset(path), **kw)

  def state(self, frames: Tensor) -> Tensor:
    """Global frame indices [...] -> float32 reference states [..., 240]."""
    return join_state(
      StateParts(
        joint_pos=self.joint_pos[frames],
        joint_vel=self.joint_vel[frames],
        body_pos=self.body_pos_w[frames],
        body_quat=self.body_quat_w[frames],
        body_lin_vel=self.body_lin_vel_w[frames],
        body_ang_vel=self.body_ang_vel_w[frames],
      )
    ).float()

  def root_state(self, frames: Tensor) -> Tensor:
    """Global frame indices [...] -> float32 floating-base states [..., 13]: position,
    quaternion (wxyz), linear and angular velocity, in the world frame."""
    return torch.cat(
      [
        self.root_pos_w[frames],
        self.root_quat_w[frames],
        self.root_lin_vel_w[frames],
        self.root_ang_vel_w[frames],
      ],
      -1,
    ).float()

  def clip_end(self, frames: Tensor) -> Tensor:
    """Global index of the last frame of each frame's clip."""
    clip = self.frame_clip[frames]
    return self.clip_start[clip] + self.clip_len[clip] - 1

  def remaining(self, frames: Tensor) -> Tensor:
    """Frames left in the clip after each frame (0 at the clip's last frame)."""
    return self.clip_end(frames) - frames

  def _window(self, frames: Tensor, K: int) -> Tensor:
    """Frames f, ..., f+K-1 [..., K], held at the clip's last frame."""
    window = frames.unsqueeze(-1) + torch.arange(K, device=self.device)
    return torch.minimum(window, self.clip_end(frames).unsqueeze(-1))

  def future(self, frames: Tensor, K: int) -> Tensor:
    """States of frames f, ..., f+K-1 [..., K, 240], held at the clip's last frame."""
    return self.state(self._window(frames, K))

  @property
  def has_human(self) -> bool:
    """Whether the clips came with human sidecars."""
    return self.human_pos_w is not None

  def human_future(self, frames: Tensor, K: int) -> Tensor:
    """Human poses of frames f, ..., f+K-1 [..., K, 24, 7] float32 (position, then
    quaternion wxyz), held at the clip's last frame like `future`."""
    if self.human_pos_w is None or self.human_quat_w is None:
      raise RuntimeError("library has no human data")
    window = self._window(frames, K)
    return torch.cat([self.human_pos_w[window], self.human_quat_w[window]], -1).float()

  def frame_bin(self, frames: Tensor) -> Tensor:
    """Bin id of each global frame: the last bin starting at or before it (bins tile
    the timeline contiguously)."""
    return torch.searchsorted(self.bin_start, frames, right=True) - 1

  def sample_frames_in_bins(self, bins: Tensor) -> Tensor:
    """One uniformly drawn global frame inside each given bin."""
    n = self.bin_len[bins]
    u = torch.rand(bins.shape, device=self.device)
    return self.bin_start[bins] + (u * n).long().minimum(n - 1)

  def local_stats(
    self, frame: FrameMode = "heading", chunk: int = 65536
  ) -> tuple[Tensor, Tensor]:
    """Per-dim mean and (population) std [268] of the canonical features of all frames,
    each in its own `frame`-mode canonical frame (`state_frame`), accumulated in
    float64 over chunks of `chunk` frames to bound memory."""
    total = torch.zeros(LOCAL_DIM, dtype=torch.float64, device=self.device)
    total_sq = torch.zeros_like(total)
    for start in range(0, self.num_frames, chunk):
      stop = min(start + chunk, self.num_frames)
      s = self.state(torch.arange(start, stop, device=self.device))
      x = canonicalize(s, state_frame(s, frame)).double()
      total += x.sum(0)
      total_sq += x.square().sum(0)
    mean = total / self.num_frames
    var = (total_sq / self.num_frames - mean.square()).clamp(min=0.0)
    return mean.float(), var.sqrt().float()
