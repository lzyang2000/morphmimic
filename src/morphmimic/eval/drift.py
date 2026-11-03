"""Drift diagnostic: how far a policy's anchor wanders from the
reference anchor over whole clips, in xy and in heading.

The benchmark (`benchmark.run_benchmark`) scores each clip by its return; a position-free
policy (goal version 2) can track the pose and the heading and still walk off the
reference's path, which the return only reflects through the global anchor-position
term. `run_drift` plays every clip of an eval env once, like the benchmark, and records
per step the xy distance between the robot anchor and the reference anchor, the
signed heading error (robot yaw minus reference yaw) and the signed height error (robot
anchor z minus reference anchor z; a jump the policy does not make shows as a dip), then
summarises each clip: mean / max / final xy error, mean / max absolute and final heading
error, min / max height error, and the time until the xy error first exceeds
`XY_THRESHOLD` and the heading error `YAW_THRESHOLD_DEG` (NaN when never).

After `env.step` the command already shows the next frame, so each sample compares the
robot's state with the reference one frame (20 ms) ahead; at walking speed that is
about 2 cm, well below the thresholds. The eval env does not auto-reset, so the last
sample of a failed clip is the state at its failing step.

`DriftTrace` keeps the per-step series on the GPU; the benchmark
(`benchmark.run_benchmark`) uses it and `summarize_drift` too, so one benchmark run also
gives these per-clip columns (`DRIFT_SUMMARY_COLUMNS`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from morphmimic.constants import GOAL_DIM, HIST_LEN, HUMAN_ENV_DIM, PROPRIO_DIM
from morphmimic.env.commands import MultiMotionCommand
from morphmimic.env.env_cfg import COMMAND_NAME
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.eval.policy_loader import PolicyFn
from morphmimic.train.collector import GOAL_GROUPS
from morphmimic.train.obs_history import ObsHistory

XY_THRESHOLD = 0.3
"""xy error (m) whose first crossing `summarize_drift` times (`t_xy_over`)."""
YAW_THRESHOLD_DEG = 15.0
"""Heading error (deg) whose first crossing `summarize_drift` times (`t_yaw_over`)."""
DRIFT_SUMMARY_COLUMNS = (
  "mean_xy",
  "max_xy",
  "final_xy",
  "mean_abs_yaw_deg",
  "max_abs_yaw_deg",
  "final_yaw_deg",
  "min_dz",
  "max_dz",
  "t_xy_over",
  "t_yaw_over",
)
"""The `summarize_drift` keys, in order: the drift columns of the drift and benchmark
tables."""
DRIFT_COLUMNS = ("name", "frames", "failed", "steps", *DRIFT_SUMMARY_COLUMNS)
"""Per clip: name, frame count, whether a termination fired, steps played, and the
`summarize_drift` keys."""
PROGRESS_EVERY = 1000


def _yaw(quat_wxyz: Tensor) -> Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def drift_errors(
  robot_pos_w: Tensor, robot_quat_w: Tensor, ref_pos_w: Tensor, ref_quat_w: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
  """xy distance [...] between robot and reference anchors, the signed heading error
  [...] (robot yaw minus reference yaw, rad, wrapped to [-pi, pi)) and the signed height
  error [...] (robot z minus reference z); quaternions wxyz."""
  diff = robot_pos_w - ref_pos_w
  xy = torch.linalg.norm(diff[..., :2], dim=-1)
  d = _yaw(robot_quat_w) - _yaw(ref_quat_w)
  yaw = torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi
  return xy, yaw, diff[..., 2]


def summarize_drift(
  xy: np.ndarray, yaw: np.ndarray, dz: np.ndarray, dt: float
) -> dict[str, float]:
  """Per-clip summary of the series `xy` (m), `yaw` (rad) and `dz` (m) sampled every
  `dt` s; `t_*_over` is the time of the first sample over the threshold, NaN when none."""
  abs_yaw_deg = np.degrees(np.abs(yaw))

  def first_over(mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(idx[0] * dt) if idx.size else math.nan

  return {
    "mean_xy": float(xy.mean()),
    "max_xy": float(xy.max()),
    "final_xy": float(xy[-1]),
    "mean_abs_yaw_deg": float(abs_yaw_deg.mean()),
    "max_abs_yaw_deg": float(abs_yaw_deg.max()),
    "final_yaw_deg": float(np.degrees(yaw[-1])),
    "min_dz": float(dz.min()),
    "max_dz": float(dz.max()),
    "t_xy_over": first_over(xy > XY_THRESHOLD),
    "t_yaw_over": first_over(abs_yaw_deg > YAW_THRESHOLD_DEG),
  }


class DriftTrace:
  """Per-step `drift_errors` of every env's running episode (the command's robot anchor
  against its reference anchor), kept on the GPU until `take` copies the finished
  episodes out."""

  def __init__(self, command: MultiMotionCommand) -> None:
    self.command = command
    # [xy, yaw, dz] x env x step; an episode lasts at most clip_len - 1 steps.
    max_len = int(command.library.clip_len.max())
    self.buf = torch.zeros(3, command.num_envs, max_len, device=command.device)

  def record(self, ep_steps: Tensor) -> None:
    """Stores this step's errors of every env at index `ep_steps` (the steps its
    episode played before this one)."""
    c = self.command
    errors = torch.stack(
      drift_errors(
        c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.anchor_pos_w, c.anchor_quat_w
      )
    )
    self.buf.scatter_(2, ep_steps.expand(3, -1).unsqueeze(-1), errors.unsqueeze(-1))

  def take(
    self, env_ids: Tensor, steps: Tensor
  ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The (xy, yaw, dz) series of the episodes of `env_ids`, the first `steps` samples
    each, in one copy to the CPU."""
    lengths = steps.tolist()
    if not lengths:
      return []
    buf = self.buf[:, env_ids, : max(lengths)].cpu().numpy()
    return [
      (buf[0, i, :n], buf[1, i, :n], buf[2, i, :n]) for i, n in enumerate(lengths)
    ]


@torch.no_grad()
def run_drift(
  env: MorphMimicEnv,
  policy_fn: PolicyFn,
  hist_len: int = HIST_LEN,
  verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
  """Plays every clip of `env`'s library once with `policy_fn` (as
  `benchmark.run_benchmark`) and returns one row per clip, in library order, with
  DRIFT_COLUMNS, plus each clip's per-step (xy, yaw, dz) series by name."""
  command = env.command_manager.get_term(COMMAND_NAME)
  assert isinstance(command, MultiMotionCommand)
  if command.cfg.sampling != "start":
    raise ValueError("run_drift needs the eval env (sampling='start').")
  lib = command.library
  device = env.device
  num_envs, num_clips = env.num_envs, lib.num_clips
  dt = float(env.step_dt)

  env_clip = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  trace = DriftTrace(command)
  history = ObsHistory(num_envs, hist_len, PROPRIO_DIM, device)
  goals = {
    "goal": torch.zeros(num_envs, GOAL_DIM, device=device),
    "goal_kp": torch.zeros(num_envs, HUMAN_ENV_DIM, device=device),
  }
  failed = torch.zeros(num_clips, dtype=torch.bool, device=device)
  series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
  next_clip = 0

  def start(env_ids: Tensor) -> None:
    nonlocal next_clip
    clips = next_clip + torch.arange(env_ids.numel(), device=device)
    next_clip += env_ids.numel()
    env_clip[env_ids] = torch.where(clips < num_clips, clips, -1)
    command.assign_clips(env_ids, clips.clamp(max=num_clips - 1))
    obs, _ = env.reset(env_ids=env_ids)
    history.reset(env_ids, obs["policy"][env_ids])
    for g in GOAL_GROUPS:
      goals[g][env_ids] = obs[g][env_ids]
    ep_steps[env_ids] = 0

  start(torch.arange(num_envs, device=device))
  step = 0
  while len(series) < num_clips:
    action = policy_fn(history.flat(), goals)
    obs, _, terminated, truncated, _ = env.step(action)
    trace.record(ep_steps)
    ep_steps += 1
    history.push(obs["policy"])
    goals = {g: obs[g].clone() for g in GOAL_GROUPS}
    step += 1

    finished = (terminated | truncated).nonzero().squeeze(-1)
    if finished.numel() > 0:
      ids = finished[env_clip[finished] >= 0]
      clips = env_clip[ids]
      failed[clips] = terminated[ids]
      series.update(zip(clips.tolist(), trace.take(ids, ep_steps[ids])))
      if len(series) < num_clips:
        start(finished)
    if verbose and step % PROGRESS_EVERY == 0:
      print(f"[morphmimic] drift step {step}: {len(series)}/{num_clips} clips", flush=True)

  rows = []
  by_name: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
  for c, name in enumerate(lib.clip_names):
    xy_c, yaw_c, dz_c = series[c]
    by_name[name] = (xy_c, yaw_c, dz_c)
    rows.append(
      {
        "name": name,
        "frames": int(lib.clip_len[c]),
        "failed": bool(failed[c]),
        "steps": len(xy_c),
        **summarize_drift(xy_c, yaw_c, dz_c, dt),
      }
    )
  return pd.DataFrame(rows, columns=list(DRIFT_COLUMNS)), by_name
