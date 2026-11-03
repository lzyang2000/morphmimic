"""Simulator data collection (Algorithm A.1 step 1).

The current policy acts on the stacked noisy `policy` obs (what the deployed policy
sees) and the goals (`GOAL_GROUPS`: the motion goal and the keypoint goal); the env
executes the action plus N(0, σ²) noise; every step is logged to the replay with the
clean obs (imagination re-adds fresh sensor noise). The env runs with auto_reset=False:
after a step, finished envs get a terminal row with the physical terminal state the step
returned, then are reset by the collector.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from morphmimic.constants import HIST_LEN, PROPRIO_DIM, R_MAX
from morphmimic.env.actions import CanonicalJointPositionAction
from morphmimic.env.commands import MultiMotionCommand
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.train.obs_history import ObsHistory
from morphmimic.train.replay import ReplayBuffer

GOAL_GROUPS = ("goal", "goal_kp")
"""Goal observation groups the policy function is given, keyed by group name."""
OBS_GROUPS = ("policy", "policy_clean", *GOAL_GROUPS, "state", "context")

PolicyFn = Callable[[Tensor, dict[str, Tensor]], Tensor]
"""(obs history [N, H*93] as `ObsHistory.flat`, goals {group: [N, dim]}) -> action."""


class Collector:
  """Steps `env` with `policy_fn(obs_hist_flat [N, H*93], goals) -> [N, 29]` (called
  under no_grad; `goals` maps each of `GOAL_GROUPS` to its obs: "goal" [N, 670] and
  "goal_kp" [N, 1684], the raw human poses then the robot pelvis quaternion) plus
  N(0, action_noise_std²) noise and logs to `replay`.

  Per step, before `env.step`, it reads the frame f_t = `command.global_frame`; row t
  of each env holds the `state` and `policy_clean` obs it acted on, the executed
  action, the reward / failure of the step, a time-out flag (`truncated` and not
  `terminated`), the `context` obs, f_t and its episode step. Envs that finished then
  get a terminal row with the returned (terminal) `state` / `policy_clean` / `context`,
  the post-step frame and episode step + 1; then they are reset and their history is
  refilled with the post-reset noisy obs.

  `on_reset(env_ids)`, if given, is called with the ids of every reset the collector
  performs (all envs at the end of `reset_all`, the finished ones at the end of a
  step's reset), once the reset is complete and before the next `policy_fn` call: the
  hook for per-episode state of the policy function.

  `obs` holds the current observation groups (what the next step acts on), `history`
  the stacked noisy obs, `ep_step` / `ep_return` the per-env step count and reward sum
  of the running episodes.
  """

  def __init__(
    self,
    env: MorphMimicEnv,
    replay: ReplayBuffer,
    policy_fn: PolicyFn,
    action_noise_std: float = 0.1,
    hist_len: int = HIST_LEN,
    on_reset: Callable[[Tensor], None] | None = None,
    det_frac: float = 0.0,
    clean_action_record: bool = False,
  ):
    """`det_frac`: the first round(det_frac * N) envs act without the exploration noise
    (`--collect-det-frac`), so the replay also holds the states the deterministic
    policy reaches. `clean_action_record` (`--clean-action-record`): the env's action
    term executes a + eps and records the clean a (mjlab's `last_action`, so the
    `prev_action` observation and the action-rate penalty), and the replay logs the
    executed action; off, the env sees and records the noisy action."""
    if replay.num_envs != env.num_envs:
      raise ValueError(
        f"replay holds {replay.num_envs} envs, the env has {env.num_envs}"
      )
    self.env = env
    self.replay = replay
    self.policy_fn = policy_fn
    self.action_noise_std = action_noise_std
    self.on_reset = on_reset
    self.noise_scale = torch.ones(env.num_envs, 1, device=env.device)
    self.action_term: CanonicalJointPositionAction | None = None
    self.set_exploration(det_frac, clean_action_record)
    command = env.command_manager.get_term(env.motion_command_name)
    assert isinstance(command, MultiMotionCommand)
    self.command = command
    n, device = env.num_envs, env.device
    self.history = ObsHistory(n, hist_len, PROPRIO_DIM, device)
    self.ep_step = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_return = torch.zeros(n, device=device)
    self.obs: dict[str, Tensor] = {}

  def set_exploration(self, det_frac: float, clean_action_record: bool) -> None:
    """Sets which envs explore (`noise_scale`: 0 for the first round(det_frac * N))
    and where the noise is added (the constructor's `det_frac` and
    `clean_action_record`); a resume applies a checkpoint's values."""
    if not 0.0 <= det_frac <= 1.0:
      raise ValueError(f"det_frac must be in [0, 1], got {det_frac}")
    n, device = self.env.num_envs, self.env.device
    num_det = round(det_frac * n)
    self.noise_scale = (torch.arange(n, device=device) >= num_det).float()[:, None]
    term = self.env.action_manager.get_term("joint_pos")
    assert isinstance(term, CanonicalJointPositionAction)
    if clean_action_record:
      term.noise_std = self.action_noise_std
      term.noise_scale.copy_(self.noise_scale)
      self.action_term = term
    else:
      term.noise_std = 0.0
      term.noise_scale.fill_(1.0)
      self.action_term = None

  def reset_all(self) -> None:
    """Reset every env, refill the histories and restart the episode counters. An
    episode cut short here gets no terminal row (its last rows just end; the next row
    starts at ep_step 0, so windows and histories do not cross it)."""
    obs, _ = self.env.reset()
    self.obs = {group: obs[group].clone() for group in OBS_GROUPS}
    self.history.reset(None, self.obs["policy"])
    self.ep_step.zero_()
    self.ep_return.zero_()
    if self.on_reset is not None:
      self.on_reset(torch.arange(self.env.num_envs, device=self.env.device))

  @torch.no_grad()
  def step(self, n: int) -> dict[str, float | int]:
    """Run n env steps for all envs. Returns stats of these steps: `mean_reward` per
    env step, `episodes_finished`, `failures` (terminated episodes), and over the
    finished episodes `mean_episode_length` (rewarded steps) and `mean_norm_return`
    (Σr / (R_MAX · length)); NaN when no episode finished. `steps` = n · num_envs."""
    if not self.obs:
      raise RuntimeError("call reset_all() before step()")
    num_envs = self.env.num_envs
    device = self.env.device
    reward_sum = torch.zeros((), device=device)
    length_sum = torch.zeros((), dtype=torch.long, device=device)
    norm_return_sum = torch.zeros((), device=device)
    failures = torch.zeros((), dtype=torch.long, device=device)
    finished = 0

    for _ in range(n):
      obs = self.obs
      frame = self.command.global_frame.clone()
      goals = {group: obs[group] for group in GOAL_GROUPS}
      action = self.policy_fn(self.history.flat(), goals)
      if self.action_term is None:  # The env sees (and records) the noisy action.
        action = action + torch.randn_like(action) * (
          self.action_noise_std * self.noise_scale
        )
      next_obs, reward, terminated, truncated, _ = self.env.step(action)
      executed = action if self.action_term is None else self.action_term.executed
      # The env reuses its buffers; keep this step's values past the reset below.
      reward, terminated = reward.clone(), terminated.clone()
      timeout = truncated & ~terminated
      self.replay.add_step(
        state=obs["state"],
        obs=obs["policy_clean"],
        action=executed,
        reward=reward,
        terminated=terminated,
        timeout=timeout,
        ctx=obs["context"],
        frame=frame,
        ep_step=self.ep_step,
      )
      self.ep_step += 1
      self.ep_return += reward
      reward_sum += reward.sum()
      self.history.push(next_obs["policy"])
      self.obs = {group: next_obs[group].clone() for group in OBS_GROUPS}

      done_ids = (terminated | timeout).nonzero().squeeze(-1)
      if done_ids.numel() == 0:
        continue
      self.replay.add_terminal_rows(
        done_ids,
        state=self.obs["state"][done_ids],
        obs=self.obs["policy_clean"][done_ids],
        ctx=self.obs["context"][done_ids],
        frame=self.command.global_frame[done_ids],
        ep_step=self.ep_step[done_ids],
      )
      length = self.ep_step[done_ids]
      finished += done_ids.numel()
      length_sum += length.sum()
      norm_return_sum += (self.ep_return[done_ids] / (R_MAX * length)).sum()
      failures += terminated.sum()

      reset_obs, _ = self.env.reset(env_ids=done_ids)
      for group in OBS_GROUPS:
        self.obs[group][done_ids] = reset_obs[group][done_ids]
      self.history.reset(done_ids, self.obs["policy"][done_ids])
      self.ep_step[done_ids] = 0
      self.ep_return[done_ids] = 0.0
      if self.on_reset is not None:
        self.on_reset(done_ids)

    steps = n * num_envs
    nan = float("nan")
    return {
      "mean_reward": reward_sum.item() / steps if steps else nan,
      "episodes_finished": finished,
      "mean_episode_length": length_sum.item() / finished if finished else nan,
      "mean_norm_return": norm_return_sum.item() / finished if finished else nan,
      "failures": int(failures.item()),
      "steps": steps,
    }
