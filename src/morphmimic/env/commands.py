"""Library-backed motion command with return-based reset sampling.

`MultiMotionCommand` is duck-type compatible with mjlab's tracking `MotionCommand`
(same properties, `update_relative_body_poses`, RSI reset), so the reward, termination
and observation functions of `mjlab.tasks.tracking.mdp` work unchanged. Each env plays
one global frame of a `MotionLibrary`.

Frames: at observation time t the command shows frame f_t; the transition's reward and
termination compare s_{t+1} with f_t (see `MorphMimicEnv`); then the frame advances.
Like mjlab, the reset path also advances once, so after a reset onto frame f the robot
holds the reference state of f and the command shows f + 1. The frame never runs past
its clip: it holds at the clip's last frame, where the env's `clip_end` time-out
(`morphmimic.env.terminations.clip_end`) ends the episode. New starts are only sampled
in `reset`, so the robot is never moved in the middle of an episode.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from torch import Tensor

from morphmimic.constants import GOAL_K, R_MAX, GoalAnchor
from morphmimic.data.motion_library import MotionLibrary
from morphmimic.robots import get_robot

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

  from morphmimic.robots import Translator

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def normalized_return(
  disc_return: Tensor, gamma_pow: Tensor, gamma: float, r_max: float
) -> Tensor:
  """Horizon-normalised episode return: the discounted return of the T steps actually
  run, `sum_{t<T} gamma^t r_{t+1}`, over the most attainable in a horizon of H steps,
  `r_max (1 - gamma^H) / (1 - gamma)`, with `gamma_pow = gamma^H` (H = T, or the
  counterfactual horizon H >= T of `MultiMotionCommandCfg.reset_norm`). At most 1
  (perfect tracking without penalties); 0 where H = 0.
  """
  attainable = r_max * (1.0 - gamma_pow) / (1.0 - gamma)
  ratio = disc_return / attainable.clamp(min=1e-12)
  return torch.where(attainable > 0.0, ratio, torch.zeros_like(ratio))


def cap_distribution(p: Tensor, pmax: float) -> Tensor:
  """`p` [n] (a distribution) with every entry capped at `pmax` and the excess spread
  over the uncapped entries in proportion to them (water-filling): the m largest
  entries sit at `pmax`, the rest are scaled by (mass - m pmax) / their mass, with m the
  smallest count for which the scaled remainder stays under the cap. Exact, and no
  host sync (a sort and a cumsum). Needs n * pmax >= the mass of `p`."""
  n = p.numel()
  sorted_p, order = torch.sort(p, descending=True)
  m = torch.arange(n + 1, device=p.device, dtype=p.dtype)
  cum = torch.cat([p.new_zeros(1), torch.cumsum(sorted_p, 0)])  # Mass of the m largest.
  rest = (cum[-1] - cum).clamp_min(1e-12)  # Mass of the others.
  scale = (cum[-1] - m * pmax) / rest
  next_largest = torch.cat([sorted_p, p.new_zeros(1)])  # The (m + 1)-th largest.
  m_star = torch.argmax((scale * next_largest <= pmax).to(torch.int8))  # First True.
  idx = torch.arange(n, device=p.device)
  out_sorted = torch.where(
    idx < m_star, torch.full_like(sorted_p, pmax), sorted_p * scale[m_star]
  )
  out = torch.empty_like(p)
  out[order] = out_sorted
  return out


def return_sampling_probs(
  g: Tensor,
  past_softmax: Tensor,
  temp: float,
  mix: tuple[float, float, float],
  cap: float | None = None,
) -> tuple[Tensor, Tensor]:
  """Reset-bin distribution of return-based sampling (App. D.1).

  Mixture of uniform over bins, `softmax(-temp * g)` of the normalised bin returns `g`
  (low-return bins first; i.e. G_max = 1) and the normalized `past_softmax`, weighted
  by `mix`. With `cap` (>= 1) no bin's share of the softmax term exceeds `cap` times
  the uniform share, the excess going to the other bins (`cap_distribution`), so
  hard-example mining cannot pile onto a few unlearnable bins. Returns (probs,
  softmax), both [num_bins].
  """
  softmax = torch.softmax(-temp * g, dim=0)
  if cap is not None:
    if cap < 1.0:
      raise ValueError(f"cap must be >= 1 (the uniform share), got {cap}")
    softmax = cap_distribution(softmax, cap / g.numel())
  probs = (
    mix[0] / g.numel() + mix[1] * softmax + mix[2] * past_softmax / past_softmax.sum()
  )
  return probs, softmax


def fold_returns(
  g: Tensor, bins: Tensor, returns: Tensor, weights: Tensor, alpha: float
) -> Tensor:
  """EMA update of per-bin returns `g` [num_bins] with finished episodes.

  Episode i (start bin `bins[i]`, normalised return `returns[i]`) counts when
  `weights[i]` is 1. A bin hit by n episodes with mean return m becomes
  `(1 - alpha)^n g + (1 - (1 - alpha)^n) m`: n EMA steps towards m, independent of
  episode order, so duplicate bins are well defined. Returns a new tensor.
  """
  count = torch.zeros_like(g).index_add_(0, bins, weights)
  total = torch.zeros_like(g).index_add_(0, bins, weights * returns)
  keep = (1.0 - alpha) ** count
  return keep * g + (1.0 - keep) * total / count.clamp(min=1.0)


def ghost_qpos(
  lib: MotionLibrary,
  frames: Tensor,
  env_origins: Tensor,
  translator: Translator,
  default_joint_pos: Tensor,
) -> Tensor:
  """Reference pose of `frames` [N] as the robot's qpos [N, 7 + J]: the floating base
  (library root + env origin, wxyz quaternion), then the canonical joints sent to the
  native hinges (`translator.to_native`; untracked joints at `default_joint_pos`)."""
  root_pos = lib.root_pos_w[frames].float() + env_origins
  root_quat = lib.root_quat_w[frames].float()
  joint_pos = translator.to_native(lib.joint_pos[frames].float(), default_joint_pos)
  return torch.cat([root_pos, root_quat, joint_pos], dim=-1)


class MultiMotionCommand(CommandTerm):
  """Motion command over a multi-clip `MotionLibrary` (see module docstring).

  `global_frame` (LongTensor [num_envs]) is the library frame each env's command
  shows; it is the collector's source of the motion frame (there is no motion-frame
  observation). `start_bin` is the reset bin each env's episode was sampled from.

  Return-based reset sampling: `G_b` [num_bins] is the EMA of the horizon-normalised
  return (`normalized_return`) of the episodes that started in each bin, so bins are
  comparable whether the episode ran the full time limit or stopped at a nearby clip
  end. The horizon is set by `cfg.reset_norm`: "remaining" (default) uses the
  counterfactual horizon T_rem fixed at the episode's start (`horizon_pow` =
  gamma^T_rem), so an episode that fails early scores low; "actual" uses the steps the
  episode ran (`gamma_pow`), so it scores by its tracking before the failure. Each
  return is clamped to >= 0 before it is folded in (the penalty terms can make it
  negative), so `G_b` stays in [0, 1], the range of the paper's `G_b / G_max`. Resets
  sample bins from `return_sampling_probs(G_b, ...)`.

  `reset` returns the per-episode means of `metrics` as 0-dim device tensors (mjlab's
  `CommandTerm.reset` converts each with `.item()`, a host sync per metric and reset).

  Robot-agnostic: `spec` is the robot's `RobotSpec` (`cfg.robot_name`); the library
  and `robot_joint_pos` / `robot_joint_vel` are canonical (29 joints), and the reset
  writes them to the native joints through `spec.translator`.
  """

  cfg: MultiMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.spec = get_robot(cfg.robot_name)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    if not cfg.subset_yaml:
      raise ValueError("MultiMotionCommandCfg.subset_yaml is not set.")
    self.library = MotionLibrary.from_yaml(
      cfg.subset_yaml,
      device=self.device,
      dtype=getattr(torch, cfg.library_dtype),
      body_names=cfg.body_names,
      robot=self.spec,
      human_root=cfg.human_root,
    )
    self.global_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Clip played by each env in sampling="start" mode (see assign_clips).
    self.env_clip = (
      torch.arange(self.num_envs, device=self.device) % self.library.num_clips
    )

    num_bodies = len(cfg.body_names)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, num_bodies, 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, num_bodies, 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Return-based reset sampling (App. D.1): per-bin EMA of the normalised return of
    # episodes that started in the bin, and EMA of past softmax weights. Per env: the
    # discounted return, gamma^T and the number T of rewarded steps of the episode,
    # and gamma^T_rem of its counterfactual horizon (set in _resample_command).
    num_bins = self.library.num_bins
    self.G_b = torch.zeros(num_bins, device=self.device)
    self.past_softmax = torch.full((num_bins,), 1.0 / num_bins, device=self.device)
    # Pre-failure restarts (`_mark_failure_restarts`): the frame each env restarts at
    # on its next resample (-1: a global start) and its run of local restarts.
    if not 0.0 <= cfg.failure_reset <= 1.0:
      raise ValueError(f"failure_reset must be in [0, 1], got {cfg.failure_reset}")
    if not 0 <= cfg.failure_jitter < cfg.failure_lookback:
      raise ValueError("failure_jitter must be in [0, failure_lookback)")
    self.restart_frame = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.consecutive_local = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.disc_return = torch.zeros(self.num_envs, device=self.device)
    self.gamma_pow = torch.ones(self.num_envs, device=self.device)
    self.horizon_pow = torch.ones(self.num_envs, device=self.device)
    self.episode_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.start_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    for name in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
      "G_mean",
      "sampling_entropy",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

  # Reference (library) quantities at the current frame. Positions are world-frame
  # (library frame + env origin), as in mjlab.

  @property
  def command(self) -> Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> Tensor:
    return self.library.joint_pos[self.global_frame].float()

  @property
  def joint_vel(self) -> Tensor:
    return self.library.joint_vel[self.global_frame].float()

  @property
  def body_pos_w(self) -> Tensor:
    return (
      self.library.body_pos_w[self.global_frame].float()
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> Tensor:
    return self.library.body_quat_w[self.global_frame].float()

  @property
  def body_lin_vel_w(self) -> Tensor:
    return self.library.body_lin_vel_w[self.global_frame].float()

  @property
  def body_ang_vel_w(self) -> Tensor:
    return self.library.body_ang_vel_w[self.global_frame].float()

  @property
  def anchor_pos_w(self) -> Tensor:
    return (
      self.library.body_pos_w[self.global_frame, self.motion_anchor_body_index].float()
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> Tensor:
    return self.library.body_quat_w[
      self.global_frame, self.motion_anchor_body_index
    ].float()

  @property
  def anchor_lin_vel_w(self) -> Tensor:
    return self.library.body_lin_vel_w[
      self.global_frame, self.motion_anchor_body_index
    ].float()

  @property
  def anchor_ang_vel_w(self) -> Tensor:
    return self.library.body_ang_vel_w[
      self.global_frame, self.motion_anchor_body_index
    ].float()

  # Robot quantities (joints canonical).

  @property
  def robot_joint_pos(self) -> Tensor:
    return self.spec.translator.to_canonical(self.robot.data.joint_pos)

  @property
  def robot_joint_vel(self) -> Tensor:
    return self.spec.translator.to_canonical_delta(self.robot.data.joint_vel)

  @property
  def robot_body_pos_w(self) -> Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  # Library-frame (no env origin) reference states.

  def ref_state(self) -> Tensor:
    """Reference state of the current frame [B, 240], library frame."""
    return self.library.state(self.global_frame)

  def future_states(self) -> Tensor:
    """Reference states of the next `goal_k` frames [B, K, 240] (held at clip end)."""
    return self.library.future(self.global_frame, self.cfg.goal_k)

  def human_future(self) -> Tensor:
    """Human poses of the next `goal_k` frames [B, K, 24, 7] (position, then quaternion
    wxyz; held at clip end), library frame. The library must have human sidecars."""
    return self.library.human_future(self.global_frame, self.cfg.goal_k)

  def assign_clips(self, env_ids: Tensor, clip_ids: Tensor) -> None:
    """Sets the clip that `env_ids` play in sampling="start" mode (from the next
    reset on)."""
    self.env_clip[env_ids] = clip_ids.to(self.device)

  # Reset sampling.

  def _sample_start(self, env_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Start frames of `env_ids` and the bins they were sampled from."""
    n = len(env_ids)
    lib = self.library
    if self.cfg.sampling == "start":
      frames = lib.clip_start[self.env_clip[env_ids]]
      bins = lib.frame_bin(frames)
    elif self.cfg.sampling == "uniform":
      frames = torch.randint(0, lib.num_frames, (n,), device=self.device)
      bins = lib.frame_bin(frames)
      self.metrics["sampling_entropy"][:] = 1.0
    elif self.cfg.sampling == "return":
      probs, _ = return_sampling_probs(
        self.G_b, self.past_softmax, self.cfg.softmax_temp, self.cfg.mix,
        self.cfg.softmax_cap,
      )
      bins = torch.multinomial(probs, n, replacement=True)
      frames = lib.sample_frames_in_bins(bins)
      entropy = -(probs * (probs + 1e-12).log()).sum()
      self.metrics["sampling_entropy"][:] = (
        entropy / math.log(lib.num_bins) if lib.num_bins > 1 else 1.0
      )
    else:
      raise ValueError(f"Unknown sampling mode {self.cfg.sampling!r}.")
    if self.cfg.failure_reset > 0.0:  # Pre-failure restarts marked by `reset`.
      restart = self.restart_frame[env_ids]
      local = restart >= 0
      frames = torch.where(local, restart, frames)
      bins = torch.where(local, lib.frame_bin(restart.clamp(min=0)), bins)
      self.restart_frame[env_ids] = -1
    # Never start on a clip's last frame, so the reset path's one-frame advance stays
    # inside the clip. The episode is credited to the sampled bin either way.
    return torch.minimum(frames, lib.clip_end(frames) - 1), bins

  def _mark_failure_restarts(self, env_ids: Tensor) -> None:
    """Pre-failure restarts (`cfg.failure_reset`): of `env_ids`, an env whose episode
    just failed (terminated, not a time-out) after at least `failure_min_steps`
    steps restarts, with probability `failure_reset`, its clip `failure_lookback` ±
    `failure_jitter` frames before the failure frame (clamped to the clip), unless it
    has done so `failure_max_consecutive` times in a row; every other env draws a
    global start. `_sample_start` consumes the marks (`restart_frame`, -1 = none)."""
    cfg = self.cfg
    if cfg.failure_reset <= 0.0:
      return
    n = len(env_ids)
    # The termination manager's flags of the last step (the env's `reset_terminated`,
    # which only exists after the first step).
    terminated = self._env.termination_manager.terminated[env_ids]
    failed = terminated & (self.episode_steps[env_ids] >= cfg.failure_min_steps)
    streak = self.consecutive_local[env_ids]
    local = (
      failed
      & (streak < cfg.failure_max_consecutive)
      & (torch.rand(n, device=self.device) < cfg.failure_reset)
    )
    frames = self.global_frame[env_ids]
    jitter = torch.randint(
      -cfg.failure_jitter, cfg.failure_jitter + 1, (n,), device=self.device
    )
    clip_start = self.library.clip_start[self.library.frame_clip[frames]]
    restart = torch.maximum(frames - cfg.failure_lookback + jitter, clip_start)
    self.restart_frame[env_ids] = torch.where(local, restart, torch.full_like(restart, -1))
    self.consecutive_local[env_ids] = torch.where(
      local, streak + 1, torch.zeros_like(streak)
    )

  def _write_reference_state_to_sim(
    self,
    env_ids: Tensor,
    root_pos: Tensor,
    root_ori: Tensor,
    root_lin_vel: Tensor,
    root_ang_vel: Tensor,
    joint_pos: Tensor,
    joint_vel: Tensor,
  ) -> None:
    """Clip native joint positions to the soft limits and write root + joint state
    to sim (as mjlab)."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: Tensor) -> None:
    frames, bins = self._sample_start(env_ids)
    self.global_frame[env_ids] = frames
    self.start_bin[env_ids] = bins
    lib = self.library
    # Counterfactual horizon T_rem: the reset path advances to frames + 1 and the
    # clip_end time-out fires on the step that shows the clip's last frame, after
    # remaining(frames) >= 1 rewarded steps; the episode cap fires after
    # max_episode_length steps.
    horizon = lib.remaining(frames).clamp(max=self._env.max_episode_length)
    self.horizon_pow[env_ids] = self.cfg.gamma ** horizon.float()

    # Reference state with mjlab's RSI noise. The root is the robot's floating base
    # (`spec.root_body`; the T2's trunk is not tracked slot 0).
    root_pos = lib.root_pos_w[frames].float() + self._env.scene.env_origins[env_ids]
    root_ori = lib.root_quat_w[frames].float()
    root_lin_vel = lib.root_lin_vel_w[frames].float()
    root_ang_vel = lib.root_ang_vel_w[frames].float()

    ranges = torch.tensor(
      [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in _POSE_KEYS], device=self.device
    )
    rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), self.device)
    root_pos = root_pos + rand[:, 0:3]
    delta_ori = quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
    root_ori = quat_mul(delta_ori, root_ori)

    ranges = torch.tensor(
      [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in _POSE_KEYS],
      device=self.device,
    )
    rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), self.device)
    root_lin_vel = root_lin_vel + rand[:, :3]
    root_ang_vel = root_ang_vel + rand[:, 3:]

    # Canonical reference joints (+ noise) to native; untracked joints at default.
    tr = self.spec.translator
    joint_pos = lib.joint_pos[frames].float()
    joint_pos = joint_pos + sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      joint_pos.shape,
      self.device,
    )
    joint_pos = tr.to_native(joint_pos, self.robot.data.default_joint_pos[0])
    joint_vel = tr.to_native_delta(lib.joint_vel[frames].float())

    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )
    self._pending_forward = True

  def update_relative_body_poses(self) -> None:
    """Recompute `body_pos_relative_w` / `body_quat_relative_w`: the reference bodies
    moved so that the reference anchor sits at the robot anchor's xy (reference z)
    with the robot anchor's heading (as mjlab)."""
    num_bodies = len(self.cfg.body_names)
    anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(
      1, num_bodies, 1
    )

    delta_pos_w = robot_anchor_pos_w
    delta_pos_w[..., 2] = anchor_pos_w[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w
    )

  # Manager hooks.

  def reset(self, env_ids: Tensor | slice | None) -> dict[str, Tensor]:
    """Folds the finished episodes of `env_ids` into `G_b` (returns normalised over the
    `cfg.reset_norm` horizon, clamped to >= 0), then resamples them. Returns the
    metrics' means over `env_ids` (0-dim device tensors, no host sync)."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    assert isinstance(env_ids, Tensor)
    alpha = self.cfg.ema_alpha
    # Only envs with at least one rewarded step end an episode (not the first reset,
    # nor a repeated reset without steps).
    finished = (self.episode_steps[env_ids] > 0).float()
    norm_pow = (
      self.horizon_pow if self.cfg.reset_norm == "remaining" else self.gamma_pow
    )
    g = normalized_return(
      self.disc_return[env_ids], norm_pow[env_ids], self.cfg.gamma, self.cfg.r_max
    ).clamp(min=0.0)  # G_b / G_max in [0, 1] (paper); penalties can make g < 0.
    self.G_b = fold_returns(self.G_b, self.start_bin[env_ids], g, finished, alpha)
    _, softmax = return_sampling_probs(
      self.G_b, self.past_softmax, self.cfg.softmax_temp, self.cfg.mix,
      self.cfg.softmax_cap,
    )
    self.past_softmax = (1.0 - alpha) * self.past_softmax + alpha * softmax
    self._mark_failure_restarts(env_ids)  # Before the episode counters are cleared.
    self.disc_return[env_ids] = 0.0
    self.gamma_pow[env_ids] = 1.0
    self.episode_steps[env_ids] = 0

    # mjlab's CommandTerm.reset, with the metric means kept on the device.
    extras: dict[str, Tensor] = {}
    for name, value in self.metrics.items():
      extras[name] = value[env_ids].mean()
      value[env_ids] = 0.0
    self.command_counter[env_ids] = 0
    self._resample(env_ids)
    # Reset-path resamples are followed by the env's own forward(); only
    # compute-path resamples (timer expiry) need our refresh (as mjlab).
    self._pending_forward = False
    return extras

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )
    self.metrics["G_mean"][:] = self.G_b.mean()

  def _update_command(self, env_ids: Tensor | None = None) -> None:
    if env_ids is None:
      # Per-step path, after the env computed this step's reward. The reset path
      # (env_ids given, dt = 0) has no new reward.
      self.disc_return += self.gamma_pow * self._env.reward_buf
      self.gamma_pow *= self.cfg.gamma
      self.episode_steps += 1

    # Advance, holding at the clip's last frame (the env's clip_end time-out has
    # already fired for envs that were showing it).
    if env_ids is None:
      self.global_frame.copy_(
        torch.minimum(self.global_frame + 1, self.library.clip_end(self.global_frame))
      )
    else:
      frames = self.global_frame[env_ids]
      self.global_frame[env_ids] = torch.minimum(
        frames + 1, self.library.clip_end(frames)
      )

    # A resample from an expired timer in compute() writes qpos/qvel without
    # refreshing derived quantities; forward() so update_relative_body_poses reads
    # the moved robot anchor (as mjlab). Never happens with the 1e9 s resampling
    # time of the env cfg.
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """mjlab's drawing, except the ghost: mjlab puts its free joint at tracked slot 0
    and writes the (here canonical) joints straight into the hinges; this ghost uses
    `ghost_qpos` (floating base by name, native joints). Frames mode reads only the
    properties above, so it is mjlab's."""
    if self.cfg.viz.mode != "ghost":
      MotionCommand._debug_vis_impl(self, visualizer)
      return
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    if self._ghost_model is None:
      # As mjlab: only visual geoms are drawn, in the ghost colour.
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      for gi in range(self._ghost_model.ngeom):
        if (
          self._ghost_model.geom_contype[gi] != 0
          or self._ghost_model.geom_conaffinity[gi] != 0
        ):
          self._ghost_model.geom_rgba[gi, 3] = 0
        else:
          self._ghost_model.geom_rgba[gi] = self._ghost_color

    indexing = self.robot.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()
    ghost = ghost_qpos(
      self.library,
      self.global_frame,
      self._env.scene.env_origins,
      self.spec.translator,
      self.robot.data.default_joint_pos[0],
    ).cpu().numpy()
    for batch in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:7]] = ghost[batch, :7]
      qpos[joint_q_adr] = ghost[batch, 7:]
      visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  subset_yaml: str
  """Subset yaml (`manifest.build_subset`); the library is built with the env."""
  robot_name: str
  """Registry name of the robot (`morphmimic.robots.get_robot`)."""
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  sampling: Literal["return", "uniform", "start"] = "return"
  gamma: float = 0.99
  r_max: float = R_MAX
  """Largest per-step reward; normalises episode returns (`normalized_return`)."""
  ema_alpha: float = 0.05
  softmax_temp: float = 25.0
  mix: tuple[float, float, float] = (0.5, 0.4, 0.1)
  """Weights of (uniform, return softmax, past softmax) in the reset-bin mixture."""
  softmax_cap: float | None = None
  """Cap on a bin's share of the return softmax term, as a multiple (>= 1) of the
  uniform share (`return_sampling_probs(cap=)`); the EMA `past_softmax` inherits it.
  None: no cap."""
  failure_reset: float = 0.0
  """Probability that an env whose episode failed after >= `failure_min_steps` steps
  restarts `failure_lookback` ± `failure_jitter` frames before the failure instead of
  drawing a global start (`_mark_failure_restarts`); at most `failure_max_consecutive`
  times in a row. 0: off."""
  failure_lookback: int = 8
  failure_jitter: int = 4
  failure_min_steps: int = 8
  failure_max_consecutive: int = 3
  reset_norm: Literal["actual", "remaining"] = "remaining"
  """Horizon over which `reset` normalises an episode's return for `G_b`. "remaining":
  the counterfactual horizon fixed at the episode's start, min(episode cap, transitions
  until the clip's last frame), whatever ends the episode, so a failure scores low and
  an episode that runs to its clip end or the cap scores as under "actual". "actual":
  the steps the episode ran, so an early failure that tracked well scores high."""
  library_dtype: str = "float32"
  human_root: str | None = None
  """Root of the clips' human sidecars (`human_convert.sidecar_path`); None: the
  default `<clip root>/human50`. The library has human data only if every clip has
  a sidecar."""
  goal_anchor: GoalAnchor = "reference"
  """Frame of the motion goal's anchor position (`rewards_torch.goal_features`):
  "reference" (position-free) or "robot" (goal version 1's robot-relative frame)."""
  goal_k: int = GOAL_K
  viz: MotionCommandCfg.VizCfg = field(default_factory=MotionCommandCfg.VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
    return MultiMotionCommand(self, env)
