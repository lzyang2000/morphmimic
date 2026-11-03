"""Benchmark evaluation (paper App. F).

Every clip of a subset is played exactly once from its first frame in the eval env (no
DR, no observation or reset noise, floor friction 1.0) with the deterministic policy.
The episode of a clip ends at the first training termination (anchor position /
orientation; the clip then counts as failed) or at the `clip_end` time-out on its last
frame.

Frames: the reset places the robot on frame 0 and the command shows frame 1, so the
transitions of a full clip are rewarded against frames 1 .. clip_len - 1, once each. The
normalised return is `sum r / (R_MAX * (clip_len - 1))`: 1 for perfect tracking of the
whole clip without penalties; a clip that fails early only collects the rewards up to
its failure, so it scores low.

Clips are handed out to the envs as they free up: every env plays its clip to the end,
then takes the next unplayed clip; envs left without one replay a dummy clip whose
results are ignored.

The policy acts through the motion goal or, for a human checkpoint, through the human
skeleton goal under a fixed mask (`evaluate`'s `interface`, `human_points`,
`kp_horizon`; `policy_loader.load_policy`); the benchmark is otherwise the same.

The same run also records the drift diagnostic (`drift.py`): per step the anchor xy,
heading and height error against the reference, summarised per clip into the
`drift.DRIFT_SUMMARY_COLUMNS` (a failed clip up to its failing step), as `run_drift`
gives them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from morphmimic.constants import (
  GOAL_DIM,
  GOAL_K,
  HIST_LEN,
  HUMAN_ENV_DIM,
  PROPRIO_DIM,
  R_MAX,
  GoalAnchor,
)
from morphmimic.data.manifest import load_subset
from morphmimic.env.commands import MultiMotionCommand, MultiMotionCommandCfg
from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.eval.drift import DRIFT_SUMMARY_COLUMNS, DriftTrace, summarize_drift
from morphmimic.eval.policy_loader import Interface, PolicyFn, load_policy
from morphmimic.model.human import HumanPoints
from morphmimic.train.collector import GOAL_GROUPS
from morphmimic.train.obs_history import ObsHistory

RESULT_COLUMNS = (
  "name",
  "frames",
  "failed",
  "fail_frame",
  "norm_return",
  "steps",
  *DRIFT_SUMMARY_COLUMNS,
)
"""Per clip: clip name, its frame count, whether a termination fired, the clip-local
frame the failing transition was compared against (-1 if none), the normalised return,
the number of rewarded steps played and the drift columns (`drift.summarize_drift`)."""
UNKNOWN_CATEGORY = "unknown"
PROGRESS_EVERY = 1000  # env steps between progress lines (verbose)


def make_eval_env(
  subset_yaml: str,
  num_envs: int,
  device: str,
  library_dtype: str = "float32",
  seed: int = 0,
  robot: str = "g1",
  builtin_pd: bool = False,
  joint_frame: str | None = None,
  imu_body: str | None = None,
  human_root: str | None = None,
  goal_anchor: GoalAnchor = "reference",
) -> MorphMimicEnv:
  """The eval env (`morphmimic_env_cfg(robot, eval=True)`) over the clips of
  `subset_yaml` (which must be `robot`'s); `builtin_pd`, `joint_frame`, `imu_body`,
  `human_root` and `goal_anchor` as in `morphmimic_env_cfg`."""
  cfg = morphmimic_env_cfg(
    robot,
    subset_yaml=subset_yaml,
    num_envs=num_envs,
    eval=True,
    builtin_pd=builtin_pd,
    joint_frame=joint_frame,
    imu_body=imu_body,
    human_root=human_root,
    goal_anchor=goal_anchor,
  )
  cfg.seed = seed
  command_cfg = cfg.commands[COMMAND_NAME]
  assert isinstance(command_cfg, MultiMotionCommandCfg)
  command_cfg.library_dtype = library_dtype
  return MorphMimicEnv(cfg, device=device)


@torch.no_grad()
def run_benchmark(
  env: MorphMimicEnv,
  policy_fn: PolicyFn,
  hist_len: int = HIST_LEN,
  verbose: bool = False,
) -> pd.DataFrame:
  """Plays every clip of `env`'s library once (see the module docstring) with
  `policy_fn(obs_hist_flat, goals) -> action` on the stacked `policy` obs (an
  `ObsHistory` of `hist_len` frames, refilled with the first obs at each clip start)
  and the goal obs (`GOAL_GROUPS`), and returns one row per clip, in library order, with
  RESULT_COLUMNS.

  `env` must be an eval env (`make_eval_env`); every env is reset first, so an env
  can be reused across calls.
  """
  command = env.command_manager.get_term(COMMAND_NAME)
  assert isinstance(command, MultiMotionCommand)
  if command.cfg.sampling != "start":
    raise ValueError("run_benchmark needs the eval env (sampling='start').")
  lib = command.library
  if (lib.clip_len < 2).any():
    raise ValueError("every clip needs at least 2 frames (one rewarded step)")
  device = env.device
  num_envs, num_clips = env.num_envs, lib.num_clips

  # Per env: the clip it plays (-1 while idle on a dummy clip), the reward sum and the
  # rewarded steps of its running episode.
  env_clip = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  ep_return = torch.zeros(num_envs, dtype=torch.float64, device=device)
  ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  trace = DriftTrace(command)
  history = ObsHistory(num_envs, hist_len, PROPRIO_DIM, device)
  goals = {
    "goal": torch.zeros(num_envs, GOAL_DIM, device=device),
    "goal_kp": torch.zeros(num_envs, HUMAN_ENV_DIM, device=device),
  }
  # Per clip results.
  failed = torch.zeros(num_clips, dtype=torch.bool, device=device)
  fail_frame = torch.full((num_clips,), -1, dtype=torch.long, device=device)
  norm_return = torch.zeros(num_clips, dtype=torch.float64, device=device)
  steps = torch.zeros(num_clips, dtype=torch.long, device=device)
  drift: list[dict[str, float]] = [{}] * num_clips
  rewarded = (lib.clip_len - 1).double()  # frames 1 .. clip_len - 1
  dt = float(env.step_dt)

  next_clip = 0

  def start(env_ids: torch.Tensor) -> None:
    """Gives `env_ids` the next unplayed clips (in order) and resets them onto their
    first frame; envs beyond the last clip get a dummy clip."""
    nonlocal next_clip
    clips = next_clip + torch.arange(env_ids.numel(), device=device)
    next_clip += env_ids.numel()
    env_clip[env_ids] = torch.where(clips < num_clips, clips, -1)
    command.assign_clips(env_ids, clips.clamp(max=num_clips - 1))
    obs, _ = env.reset(env_ids=env_ids)
    history.reset(env_ids, obs["policy"][env_ids])
    for g in GOAL_GROUPS:
      goals[g][env_ids] = obs[g][env_ids]
    ep_return[env_ids] = 0.0
    ep_steps[env_ids] = 0

  start(torch.arange(num_envs, device=device))
  num_done = 0
  step = 0
  while num_done < num_clips:
    frame = command.global_frame.clone()
    action = policy_fn(history.flat(), goals)
    obs, reward, terminated, truncated, _ = env.step(action)
    ep_return += reward.double()
    trace.record(ep_steps)
    ep_steps += 1
    history.push(obs["policy"])
    goals = {g: obs[g].clone() for g in GOAL_GROUPS}
    step += 1

    finished = (terminated | truncated).nonzero().squeeze(-1)
    if finished.numel() > 0:
      ids = finished[env_clip[finished] >= 0]
      clips = env_clip[ids]
      term = terminated[ids]
      failed[clips] = term
      local = frame[ids] - lib.clip_start[clips]
      fail_frame[clips] = torch.where(term, local, torch.full_like(local, -1))
      norm_return[clips] = ep_return[ids] / (R_MAX * rewarded[clips])
      steps[clips] = ep_steps[ids]
      for c, series in zip(clips.tolist(), trace.take(ids, ep_steps[ids])):
        drift[c] = summarize_drift(*series, dt)
      num_done += ids.numel()
      if num_done < num_clips:
        start(finished)
    if verbose and step % PROGRESS_EVERY == 0:
      print(f"[morphmimic] eval step {step}: {num_done}/{num_clips} clips", flush=True)

  return pd.DataFrame(
    {
      "name": lib.clip_names,
      "frames": lib.clip_len.cpu().numpy(),
      "failed": failed.cpu().numpy(),
      "fail_frame": fail_frame.cpu().numpy(),
      "norm_return": norm_return.cpu().numpy(),
      "steps": steps.cpu().numpy(),
      **{k: [d[k] for d in drift] for k in DRIFT_SUMMARY_COLUMNS},
    },
    columns=list(RESULT_COLUMNS),
  )


def evaluate(
  ckpt: str | Path,
  subset_yaml: str,
  device: str,
  num_envs: int = 4096,
  library_dtype: str = "float32",
  verbose: bool = False,
  robot: str | None = None,
  interface: Interface = "motion",
  human_points: HumanPoints = "full",
  kp_horizon: int = GOAL_K,
  human_root: str | None = None,
  human_heading: bool = True,
  builtin_pd: bool = False,
  policy_joint_frame: bool = False,
  morph_robot: str | None = None,
  morph_mix: tuple[str, str, float] | None = None,
  morph_vector: torch.Tensor | None = None,
  norm_from: str | None = None,
  imu_body: str | None = None,
  goal_anchor: GoalAnchor = "reference",
  human_drift: bool = False,
) -> pd.DataFrame:
  """Benchmark of checkpoint `ckpt` on the clips of `subset_yaml` with up to
  `num_envs` parallel envs (never more than there are clips), on `robot` (default: the
  checkpoint's robot). Returns `run_benchmark`'s per-clip table; `df.attrs` holds
  `ckpt`, `iteration`, `subset_yaml` and `mode` ("motion" or
  "human-<points>-h<horizon>", "-noyaw" added without the relative orientation).

  `interface`, `human_points`, `kp_horizon`, `human_heading` and `human_drift` select
  the goal the policy acts on, as in `load_policy` (which also rejects an invalid selection, before
  any env is built).
  `human_root` is where the env finds the clips' human sidecars (`morphmimic_env_cfg`);
  the human interface raises ValueError when the clips have none (its goal would be
  zeros).

  Transfer: `builtin_pd`, implicit PD on every joint;
  `policy_joint_frame`, the canonical joints use the action scale and default pose of
  the checkpoint's robot, not `robot`'s; `morph_robot`, the actor morphology vector of
  that robot instead of the stored one; `morph_mix=(a, b, alpha)`, the vector
  `(1 - alpha) * vec(a) + alpha * vec(b)`; `morph_vector`, a given actor vector (all
  `load_policy`); `imu_body`, gyro and
  gravity read on that tracked body (`morphmimic_env_cfg`). `goal_anchor` is the
  motion goal's frame, the checkpoint's (`load_policy` refuses another)."""
  policy_fn, meta = load_policy(
    ckpt,
    device,
    interface,
    human_points,
    kp_horizon,
    morph_robot=morph_robot,
    morph_mix=morph_mix,
    morph_vector=morph_vector,
    norm_from=norm_from,
    goal_anchor=goal_anchor,
    human_heading=human_heading,
    human_drift=human_drift,
  )
  num_envs = min(num_envs, len(load_subset(subset_yaml)))
  env = make_eval_env(
    subset_yaml,
    num_envs,
    device,
    library_dtype,
    robot=robot or meta.robot,
    builtin_pd=builtin_pd,
    joint_frame=meta.robot if policy_joint_frame else None,
    imu_body=imu_body,
    human_root=human_root,
    goal_anchor=goal_anchor,
  )
  try:
    if interface == "human":
      require_human_sidecars(env, subset_yaml, human_root)
    df = run_benchmark(env, policy_fn, meta.hist_len, verbose=verbose)
  finally:
    env.close()
  df.attrs.update(
    ckpt=meta.path, iteration=meta.iteration, subset_yaml=subset_yaml, mode=meta.mode
  )
  return df


def require_human_sidecars(
  env: MorphMimicEnv, subset_yaml: str, human_root: str | None
) -> None:
  """Raise ValueError when the env's clips have no human sidecars: the `goal_kp` group
  is then zeros, and a policy acting through the human interface would act on an
  empty goal (shared by the benchmark and `scripts.play`)."""
  command = env.command_manager.get_term(COMMAND_NAME)
  assert isinstance(command, MultiMotionCommand)
  if not command.library.has_human:
    raise ValueError(
      f"the human interface needs human sidecars; the clips of {subset_yaml}"
      f" have none (human_root {human_root or '<clip root>/human50'})"
    )


def summarize(df: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
  """Aggregate benchmark numbers of a `run_benchmark` table. Categories come from the
  manifest's `category` column (joined on `name`); clips missing from the manifest
  count as UNKNOWN_CATEGORY. Per-category dicts cover every category in `df`. `drift`
  holds the means over the clips of `mean_xy`, `final_xy` and `mean_abs_yaw_deg` and
  the number of clips whose xy / heading error crossed `drift.XY_THRESHOLD` /
  `drift.YAW_THRESHOLD_DEG` (`clips_xy_over`, `clips_yaw_over`)."""
  categories = manifest.drop_duplicates("name").set_index("name")["category"]
  category = df["name"].map(categories).fillna(UNKNOWN_CATEGORY)
  category = category.where(category != "", UNKNOWN_CATEGORY).rename("category")
  groups = df.groupby(category)
  num_clips = len(df)
  return {
    "num_clips": num_clips,
    "failures": int(df["failed"].sum()),
    "failure_rate": float(df["failed"].mean()) if num_clips else math.nan,
    "mean_norm_return": float(df["norm_return"].mean()),
    "median_norm_return": float(df["norm_return"].median()),
    "clips_by_category": {str(k): int(v) for k, v in groups.size().items()},
    "failures_by_category": {str(k): int(v) for k, v in groups["failed"].sum().items()},
    "mean_norm_return_by_category": {
      str(k): float(v) for k, v in groups["norm_return"].mean().items()
    },
    "drift": {
      "mean_xy": float(df["mean_xy"].mean()),
      "final_xy": float(df["final_xy"].mean()),
      "mean_abs_yaw_deg": float(df["mean_abs_yaw_deg"].mean()),
      "clips_xy_over": int(df["t_xy_over"].notna().sum()),
      "clips_yaw_over": int(df["t_yaw_over"].notna().sum()),
    },
  }
