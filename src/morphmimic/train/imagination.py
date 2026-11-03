"""Policy improvement through the frozen world model (paper Eq. 2, Eq. 3).

Rollouts start from replay rows (`StartBatch`): the logged state s_0, its context xi,
the clean obs history and the reference frame f_0 shown at that time. Imagination
follows the simulator's frame rules: transition t is rewarded and judged against
ref_t = state(f_t) with f_{t+1} = min(f_t + 1, clip_end), and the episode is cut after
the transition that starts at the clip's last frame (the `clip_end` time-out). So a
start with n_0 = remaining(f_0) frames left has T_b = min(H, n_0 + 1) rewarded steps;
the rollout still runs H steps (frames held at the clip end) but `trunc_mask` drops
rewards at t >= T_b and the critic bootstraps at ŝ_{T_b}.

    a_t     = pi(obs_norm(o_{t-9..t}), goal_norm(g_t)),   g_t = goal(ŝ_t, ref f_t..f_t+K-1)
    ŝ_{t+1} = f(ŝ_t, a_t, xi)
    r̂_{t+1} = r(ŝ_{t+1}, ref_t, a_t, a_{t-1}),  d̂_{t+1} = d(ŝ_{t+1}, ref_t)
    ĉ_0 = 1,  ĉ_{t+1} = ĉ_t (1 - d̂_{t+1})
    o_{t+1} = g(ŝ_{t+1}, xi, a_t) + eta  (eta: Table A.1 sensor noise)
    J = mean_b [ sum_{t<T_b} γ^t ĉ_t r̂_{t+1} + γ^{T_b} ĉ_{T_b} V(ŝ_{T_b}, g_{T_b}, xi, p_{T_b}) ]

The critic also gets the privileged p_t = critic_anchor_pos(ŝ_t, ref f_t..f_t+K-1),
the reference anchor in the robot anchor frame, which the position-free goal g_t
lacks; the policy never sees it.

The replay holds clean observations; fresh sensor noise is drawn for every history
frame (the starting ones included), all of it in one draw before the rollout
(`imagination_noise`). a_{-1} is the executed action stored in the newest history
frame. World-model and critic parameters are frozen during the rollout, so gradients
reach only the policy, through the states, observations and values.

`imagine` = noise draw + `frozen` + `imagine_rollout`: H calls of `imagine_step`,
then `finish_rollout` (stacking, values). The trainer passes compiled versions of
these two (fixed shapes, no host sync) and keeps the noise draw and the freezing
outside the compiled regions.

Human variant: `imagine_step_human` acts from the masked, jittered human skeleton
goal (the 24 SMPL joints of the K future frames, `model/human.py`) through the human
encoder, with the human Spine3's (else the anchor's) orientation relative to the
imagined torso (anchor body) orientation per future frame through its own normaliser
(keypoint goal version 10; zeros where the heading mask is off),

    z_t = enc_h(mask(human_norm(human(H_t..H_t+K-1, jitter)),
                     heading_norm(rel(q(ŝ_t), H_t..))))
    a_t = dec(z_t, obs_norm(o_{t-9..t}))
    e_t = ||z_t - sg(enc(goal_norm(g_t)))||^2

with the same world model, reward, termination and critic input (the full goal g_t,
never the human goal). Every step also records the action gap to the motion teacher
c_t = ||a_t - sg(dec(enc(goal_norm(g_t)), ...))||^2 / 29. `student_objective` sums
three terms, each weighted and divided by its running scale (`TermScales`): -J
(`return` student only), the mean of c_t and the mean of e_t over the rewarded steps.
`return` keeps the gradient through the dynamics, with its own critic. `copy` detaches
ŝ_t and the history at each step (no gradient through the dynamics) and the rollout
has no critic (`finish_rollout(critic=None)`). The jitter perturbs only the actor's
input (`human_jitter_window`, drawn once per rollout like the sensor noise); reward,
termination and goal use the clean reference.

Direct student (`direct`, the direct human policy `--policy-goal human`): the human
encoder is the policy's encoder. Nothing is detached, the motion encoder is not called
(no teacher: e_t and c_t are zeros) and every step records the reconstruction error
of the stored goal, m_t = ||recon(z'_t, morph) - sg(goal_norm(g_t))||^2 / 670
(`Policy.reconstruct`), where z'_t is z_t with the heading block of the detached
torso orientation sg(q(ŝ_t)): m_t trains the human encoder and the head, never the
decoder or earlier actions through the dynamics. `student_objective` then sums -J (with the motion critic) and
the mean of m_t over the rewarded steps, each weighted and over its scale.

The policy's and the critic's gradient clipping (`clip_grads_`, `fit_critic`) zero a
non-finite gradient instead of stepping on it (`guarded_clip_grad_norm_`).
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch
from torch import Tensor, nn

from morphmimic.constants import (
  GOAL_K,
  HUMAN_JITTER,
  NUM_JOINTS,
  OBS_NOISE,
  PROPRIO_DIM,
  GoalAnchor,
)
from morphmimic.model.human import (
  heading_features,
  heading_from_goal,
  human_features,
  human_features_from_goal,
  human_jitter,
  mask_human_goal,
)
from morphmimic.model.rewards_torch import (
  PROPRIO_SLICES,
  critic_anchor_pos,
  goal_features,
)
from morphmimic.model.state import anchor_quat
from morphmimic.train.normalizers import RunningNormalizer, normalize_history
from morphmimic.train.obs_history import ObsHistory

if TYPE_CHECKING:
  from morphmimic.data.motion_library import MotionLibrary
  from morphmimic.model.policy import Critic, Policy
  from morphmimic.model.world_model import WorldModel
  from morphmimic.train.replay import StartBatch

# Encoder bias inside the context xi = [friction 1 | encoder bias 29 | CoM offset 3].
_CTX_ENCODER_BIAS = slice(1, 1 + NUM_JOINTS)

# `imagine_step` / `finish_rollout` with the networks bound (see there).
StepFn = Callable[..., "ImaginedStep"]
FinishFn = Callable[..., "ImaginedRollout"]


@dataclass
class ImaginationCfg:
  horizon: int = 32
  gamma: float = 0.99
  lam: float = 0.95
  obs_noise: dict[str, float] = field(default_factory=lambda: dict(OBS_NOISE))
  critic_epochs: int = 5
  critic_minibatches: int = 4
  # Half-widths of the human target jitter (`HUMAN_JITTER` keys, `human_jitter`).
  human_jitter_half: dict[str, float] = field(
    default_factory=lambda: dict(HUMAN_JITTER)
  )
  clip_end_terminal: bool = False
  """Value zero after a clip's last frame (no bootstrap when the clip ends within the
  horizon). False: bootstrap V(ŝ_{T_b}) as for a time-out."""
  target_max: float | None = None
  """Upper clamp on each critic target R_t, applied inside the recursion so earlier
  targets bootstrap from the clamped ones (None: unclamped, the paper's targets)."""
  goal_anchor: GoalAnchor = "reference"
  """Frame of the motion goal's anchor position (`goal_features`), as the env's."""


class ImaginedRollout(NamedTuple):
  """H imagined steps from B starts. Network inputs are stored as the networks see
  them (normalised goal and local features, scaled context)."""

  states: Tensor  # ŝ_0 .. ŝ_H [B, H+1, 240]
  actions: Tensor  # a_0 .. a_{H-1} [B, H, 29]
  rewards: Tensor  # r̂_1 .. r̂_H [B, H]
  dones: Tensor  # termination probabilities d̂_1 .. d̂_H [B, H]
  survival: Tensor  # ĉ_0 .. ĉ_H [B, H+1]
  values: Tensor  # V_0 .. V_H [B, H+1]
  goals: Tensor  # goal_norm(g_0) .. goal_norm(g_H) [B, H+1, 670]
  priv: Tensor  # critic_anchor_pos p_0 .. p_H [B, H+1, 30]
  local: Tensor  # wm.local_features(ŝ_0 .. ŝ_H) [B, H+1, 268]
  ctx: Tensor  # wm.scale_ctx(xi) [B, 33]
  trunc_mask: Tensor  # bool, t < T_b (step t is rewarded) [B, H]
  latent_err: Tensor  # ||z_h,t - sg(z_motion,t)||^2 [B, H]; zeros if not human
  action_err: Tensor  # human student's teacher action gap [B, H]; zeros if not human
  bootstrap: Tensor | None = None  # weight of V_{T_b} [B] (0: clip ended); None = 1
  recon_err: Tensor | None = None  # direct student's goal reconstruction error [B, H]

  @property
  def num_rewarded(self) -> Tensor:
    """T_b [B] (long): rewarded steps; the critic bootstraps at ŝ_{T_b}."""
    return self.trunc_mask.sum(dim=1)


@contextlib.contextmanager
def frozen(*modules: nn.Module) -> Iterator[None]:
  """Within the block the modules' parameters do not require grad (flags restored
  after), so graphs built inside never reach them; gradients still flow through the
  modules to their inputs."""
  params = [p for m in modules for p in m.parameters()]
  flags = [p.requires_grad for p in params]
  for p in params:
    p.requires_grad_(False)
  try:
    yield
  finally:
    for p, flag in zip(params, flags):
      p.requires_grad_(flag)


def act(
  policy: Policy,
  obs_norm: RunningNormalizer,
  goal_norm: RunningNormalizer,
  obs_hist_flat: Tensor,
  goal: Tensor,
) -> Tensor:
  """Action [B, 29] from the raw stacked obs [B, H * 93] and goal [B, 670]; the policy
  input path shared by collection, evaluation and imagination."""
  return policy(normalize_history(obs_norm, obs_hist_flat), goal_norm(goal))


def act_human(
  policy: Policy,
  obs_norm: RunningNormalizer,
  human_norm: RunningNormalizer,
  heading_norm: RunningNormalizer,
  obs_hist_flat: Tensor,
  human_goal_raw: Tensor,
  point_mask: Tensor,
  frame_mask: Tensor,
  heading_mask: Tensor,
) -> Tensor:
  """`act` through the human encoder: action [B, 29] from the raw stacked obs
  [B, H * 93], the raw human goal [B, 1684] (`pack_human_goal`) and the bool masks
  `point_mask` [B, 24], `frame_mask` [B, K], `heading_mask` [B]. The features go
  through `human_norm` and the heading block (`heading_from_goal`) through
  `heading_norm`; the masks act after the normalisation (`mask_human_goal`), so the
  raw values under them do not matter."""
  feats = human_features_from_goal(human_goal_raw, point_mask)
  heading = heading_norm(heading_from_goal(human_goal_raw, point_mask))
  goal = mask_human_goal(
    human_norm(feats), point_mask, frame_mask, heading, heading_mask
  )
  return policy.forward_human(normalize_history(obs_norm, obs_hist_flat), goal)


def noise_half_widths(noise: dict[str, float], device: torch.device) -> Tensor:
  """Per-dim half-widths [93] of the uniform sensor noise (Table A.1, keys of
  `OBS_NOISE`); zero on the previous action."""
  half = torch.zeros(PROPRIO_DIM, device=device)
  for key in ("joint_pos", "joint_vel", "ang_vel", "gravity"):
    half[getattr(PROPRIO_SLICES, key)] = noise[key]
  return half


def sensor_noise(shape: tuple[int, ...], half: Tensor) -> Tensor:
  """Uniform noise [*shape, 93] in [-half, half]."""
  return (2.0 * torch.rand(*shape, half.shape[0], device=half.device) - 1.0) * half


def imagination_noise(
  start: StartBatch, cfg: ImaginationCfg, half: Tensor | None = None
) -> Tensor:
  """Sensor noise of one rollout [B, hist_len + H, 93] (Table A.1): entry j < hist_len
  for start history frame j, entry hist_len + t for the observation after step t.
  `half` defaults to `noise_half_widths(cfg.obs_noise)`."""
  b, hist_len = start.obs_hist.shape[:2]
  if half is None:
    half = noise_half_widths(cfg.obs_noise, start.obs_hist.device)
  return sensor_noise((b, hist_len + cfg.horizon), half)


def reference_window(library: MotionLibrary, frame0: Tensor, length: int) -> Tensor:
  """Reference states of frames min(f_0 + j, clip_end), j < length -> [B, length, 240].
  Entry t is ref_t, and entries t .. t+K-1 are `library.future(f_t, K)`."""
  offsets = torch.arange(length, device=frame0.device)
  end = library.clip_end(frame0).unsqueeze(-1)
  return library.state(torch.minimum(frame0.unsqueeze(-1) + offsets, end))


def human_window(library: MotionLibrary, frame0: Tensor, length: int) -> Tensor:
  """Human poses of frames min(f_0 + j, clip_end), j < length -> [B, length, 24, 7]
  (position, then quaternion), aligned with `reference_window`."""
  return library.human_future(frame0, length)


def imagine(
  wm: WorldModel,
  policy: Policy,
  critic: Critic,
  library: MotionLibrary,
  start: StartBatch,
  cfg: ImaginationCfg,
  obs_norm: RunningNormalizer,
  goal_norm: RunningNormalizer,
  noise: Tensor | None = None,
) -> ImaginedRollout:
  """H-step rollout of the policy through the world model (module docstring).

  Differentiable w.r.t. the policy parameters; the world model and critic are frozen
  for the call. The normalisers are used as constants (never updated here). `noise`
  [B, hist_len + H, 93] is the sensor noise (`imagination_noise`), drawn here if None.
  """
  if noise is None:
    noise = imagination_noise(start, cfg)
  with frozen(wm, critic):
    return imagine_rollout(
      wm, policy, critic, library, start, noise, cfg, obs_norm, goal_norm
    )


class ImaginedStep(NamedTuple):
  """One imagined transition from ŝ_t (`imagine_step`)."""

  goal: Tensor  # goal_norm(g_t) [B, 670]
  priv: Tensor  # critic_anchor_pos p_t [B, 30]
  action: Tensor  # a_t [B, 29]
  next_state: Tensor  # ŝ_{t+1} [B, 240]
  reward: Tensor  # r̂_{t+1} [B]
  done: Tensor  # d̂_{t+1} [B]
  survival: Tensor  # ĉ_{t+1} [B]
  obs_hist: Tensor  # noisy history ending with o_{t+1} + eta [B, hist_len, 93]
  latent_err: Tensor  # ||z_h,t - sg(z_motion,t)||^2 [B]; zeros for `imagine_step`
  action_err: Tensor  # human student's teacher action gap [B]; zeros for `imagine_step`
  recon_err: Tensor  # direct student's goal reconstruction error [B]; else zeros


def anchor_kwargs(anchor: GoalAnchor) -> dict[str, GoalAnchor]:
  """`anchor=` for `imagine_step`, `imagine_step_human` and `finish_rollout`: empty
  for the default "reference", so the default calls are exactly the positional ones
  of before the switch."""
  return {} if anchor == "reference" else {"anchor": anchor}


def imagine_step(
  wm: WorldModel,
  policy: Policy,
  obs_norm: RunningNormalizer,
  goal_norm: RunningNormalizer,
  state: Tensor,
  obs_hist: Tensor,
  prev_action: Tensor,
  survival: Tensor,
  goal_ref: Tensor,
  ctx: Tensor,
  noise: Tensor,
  *,
  anchor: GoalAnchor = "reference",
) -> ImaginedStep:
  """One imagined transition (module docstring) from ŝ_t with the noisy obs history
  [B, hist_len, 93], a_{t-1}, ĉ_t, `goal_ref` [B, K, 240] = ref_t .. ref_{t+K-1}, the
  context xi and the sensor noise eta [B, 93] of o_{t+1}; the goal in the `anchor`
  frame (`goal_features`). The unit the trainer compiles: one graph, replayed H times
  per rollout."""
  goal = goal_norm(goal_features(state, goal_ref, anchor))
  # Same input path as `act`.
  action = policy(normalize_history(obs_norm, obs_hist.flatten(1)), goal)
  return _transition(
    wm, state, goal, action, obs_hist, prev_action, survival, goal_ref, ctx, noise
  )


def _transition(
  wm: WorldModel,
  state: Tensor,
  goal: Tensor,
  action: Tensor,
  obs_hist: Tensor,
  prev_action: Tensor,
  survival: Tensor,
  goal_ref: Tensor,
  ctx: Tensor,
  noise: Tensor,
  latent_err: Tensor | None = None,
  action_err: Tensor | None = None,
  recon_err: Tensor | None = None,
) -> ImaginedStep:
  """The part of an imagined step after the action a_t (`imagine_step`,
  `imagine_step_human`): world-model step, reward, termination, next noisy
  observation. `latent_err`, `action_err` and `recon_err` None mean zeros."""
  next_state = wm.step(state, action, ctx)
  ref = goal_ref[:, 0]
  reward = wm.reward(next_state, ref, action, prev_action)
  done = wm.termination_prob(next_state, ref)
  obs = wm.obs(next_state, ctx[:, _CTX_ENCODER_BIAS], action)
  return ImaginedStep(
    goal=goal,
    priv=critic_anchor_pos(state, goal_ref),
    action=action,
    next_state=next_state,
    reward=reward,
    done=done,
    survival=survival * (1.0 - done),
    obs_hist=ObsHistory.shifted(obs_hist, obs + noise),
    latent_err=torch.zeros_like(reward) if latent_err is None else latent_err,
    action_err=torch.zeros_like(reward) if action_err is None else action_err,
    recon_err=torch.zeros_like(reward) if recon_err is None else recon_err,
  )


def imagine_step_human(
  wm: WorldModel,
  policy: Policy,
  obs_norm: RunningNormalizer,
  goal_norm: RunningNormalizer,
  human_norm: RunningNormalizer,
  heading_norm: RunningNormalizer,
  student: Literal["copy", "return", "direct"],
  state: Tensor,
  obs_hist: Tensor,
  prev_action: Tensor,
  survival: Tensor,
  goal_ref: Tensor,
  ctx: Tensor,
  noise: Tensor,
  point_mask: Tensor,
  frame_mask: Tensor,
  heading_mask: Tensor,
  jitter: Tensor,
  human_win: Tensor,
  *,
  anchor: GoalAnchor = "reference",
) -> ImaginedStep:
  """`imagine_step` with the action from the human encoder: the arguments after
  `noise` are the bool masks `point_mask` [B, 24], `frame_mask` [B, K] and
  `heading_mask` [B], the target jitter [B, K, 24, 6]
  (`human_jitter_window[:, t : t + K]`) and the human poses `human_win`
  [B, K, 24, 7] of the K frames of `goal_ref` (`human_window[:, t : t + K]`). The
  heading block reads the full torso orientation of `state` (the imagined state,
  world frame) against the clean human anchor (`heading_features`), through
  `heading_norm` and then the masks, as `act_human`. The goal stored for the critic
  is the full motion goal; `latent_err` [B] is the squared distance of z_h to the
  motion encoder's latent of that goal, a stop-gradient target, and `action_err` [B]
  the mean squared gap of the student's action to the decoder's action on the motion
  latent (the teacher, detached), both on the detached observation history (one
  extra decoder pass). `anchor` is the motion goal's (`goal_features`).

  `student` "copy" detaches `state` and `obs_hist` first (no gradient between steps;
  the world model still steps them with the student's action, detached, so autograd
  keeps no world-model graph and the stored action, rewards and states carry none).
  "return" detaches nothing on the action that drives the world model: the return
  term flows through the dynamics. The copy term never does (detached history), so it
  only matches the teacher at the states the student reaches; the latent pull keeps
  its path through the heading block (the imagined torso orientation), as v2's pull
  did. "direct" (the direct human policy) detaches nothing as "return" but has no
  teacher: the motion encoder is not called, `latent_err` and `action_err` are zeros,
  and `recon_err` [B] is the mean squared error of `policy.reconstruct(z_r)` to the
  stored goal (detached), z_r being z_h with the heading block built from the
  detached torso orientation (one more human-encoder pass), so the term trains the
  human encoder and the head only, never the decoder or the dynamics path. The unit the trainer compiles for the human step."""
  if student == "copy":
    state, obs_hist = state.detach(), obs_hist.detach()
  elif student not in ("return", "direct"):
    raise ValueError(
      f"Expected student 'copy', 'return' or 'direct', got {student!r}."
    )
  goal = goal_norm(goal_features(state, goal_ref, anchor))
  z_motion = None if student == "direct" else policy.encode(goal).detach()
  human_quat = human_win[..., 3:]
  feats = human_features(human_win[..., :3], human_quat, point_mask, jitter)
  robot_quat = anchor_quat(state)
  heading = heading_norm(heading_features(robot_quat, human_quat, point_mask))
  feats_n = human_norm(feats)
  z_h = policy.encode_human(
    mask_human_goal(feats_n, point_mask, frame_mask, heading, heading_mask)
  )
  hist = normalize_history(obs_norm, obs_hist.flatten(1))
  action = policy.decoder(z_h, hist)
  if z_motion is None:  # direct: no teacher, the reconstruction error instead
    # The latent of the reconstruction reads a detached torso orientation: through
    # the live heading the term would push the imagined torso (and, through the
    # dynamics, earlier actions and the decoder) toward a better reconstruction.
    heading_sg = heading_norm(
      heading_features(robot_quat.detach(), human_quat, point_mask)
    )
    z_r = policy.encode_human(
      mask_human_goal(feats_n, point_mask, frame_mask, heading_sg, heading_mask)
    )
    recon_err = ((policy.reconstruct(z_r) - goal.detach()) ** 2).mean(-1)
    return _transition(
      wm, state, goal, action, obs_hist, prev_action, survival, goal_ref, ctx, noise,
      recon_err=recon_err,
    )
  # The copy gap on a detached history: it matches the teacher at the states the
  # student reaches and sends no gradient into earlier actions.
  hist_fixed = hist.detach()
  a_teacher = policy.decoder(z_motion, hist_fixed).detach()
  action_err = ((policy.decoder(z_h, hist_fixed) - a_teacher) ** 2).mean(-1)
  if student == "copy":
    action = action.detach()
  return _transition(
    wm,
    state,
    goal,
    action,
    obs_hist,
    prev_action,
    survival,
    goal_ref,
    ctx,
    noise,
    latent_err=((z_h - z_motion) ** 2).sum(-1),
    action_err=action_err,
  )


def finish_rollout(
  wm: WorldModel,
  critic: Critic | None,
  goal_norm: RunningNormalizer,
  state0: Tensor,
  steps: list[ImaginedStep],
  last_goal_ref: Tensor,
  ctx: Tensor,
  trunc_mask: Tensor,
  bootstrap: Tensor | None = None,
  *,
  anchor: GoalAnchor = "reference",
) -> ImaginedRollout:
  """`ImaginedRollout` of the H `steps` from ŝ_0 = `state0`: the goal g_H (from
  `last_goal_ref` = ref_H .. ref_{H+K-1}), the local features of ŝ_0 .. ŝ_H and the
  critic values V_0 .. V_H (zeros with `critic` None: the copy student); g_H in the
  `anchor` frame (`goal_features`). The second unit the trainer compiles."""
  states = torch.stack([state0] + [s.next_state for s in steps], dim=1)
  goal_h = goal_norm(goal_features(steps[-1].next_state, last_goal_ref, anchor))
  goals = torch.stack([s.goal for s in steps] + [goal_h], dim=1)
  priv_h = critic_anchor_pos(steps[-1].next_state, last_goal_ref)
  priv = torch.stack([s.priv for s in steps] + [priv_h], dim=1)
  survival0 = torch.ones_like(steps[0].survival)
  b, length = states.shape[:2]
  local = wm.local_features(states)
  ctx_in = wm.scale_ctx(ctx)
  if critic is None:
    values = local.new_zeros(b, length)
  else:
    values = critic(
      local.flatten(0, 1),
      goals.flatten(0, 1),
      ctx_in.unsqueeze(1).expand(-1, length, -1).flatten(0, 1),
      priv.flatten(0, 1),
    ).view(b, length)
  return ImaginedRollout(
    states=states,
    actions=torch.stack([s.action for s in steps], dim=1),
    rewards=torch.stack([s.reward for s in steps], dim=1),
    dones=torch.stack([s.done for s in steps], dim=1),
    survival=torch.stack([survival0] + [s.survival for s in steps], dim=1),
    values=values,
    goals=goals,
    priv=priv,
    local=local,
    ctx=ctx_in,
    trunc_mask=trunc_mask,
    latent_err=torch.stack([s.latent_err for s in steps], dim=1),
    action_err=torch.stack([s.action_err for s in steps], dim=1),
    bootstrap=bootstrap,
    recon_err=torch.stack([s.recon_err for s in steps], dim=1),
  )


def human_jitter_window(
  start: StartBatch, cfg: ImaginationCfg, half: dict[str, float] | None = None
) -> Tensor:
  """Target jitter of one human rollout [B, H + K, 24, 6] (`human_jitter`): entry j
  for the reference frame of `reference_window` entry j, so step t's actor sees
  `[:, t : t + K]` and a frame keeps its jitter across the steps whose window contains
  it. `half` defaults to `cfg.human_jitter_half`."""
  if half is None:
    half = cfg.human_jitter_half
  b = start.state.shape[0]
  return human_jitter((b, cfg.horizon + GOAL_K), half, start.state.device)


def imagine_rollout(
  wm: WorldModel,
  policy: Policy,
  critic: Critic,
  library: MotionLibrary,
  start: StartBatch,
  noise: Tensor,
  cfg: ImaginationCfg,
  obs_norm: RunningNormalizer,
  goal_norm: RunningNormalizer,
  step: StepFn | None = None,
  finish: FinishFn | None = None,
  extras: Callable[[int], tuple[Tensor, ...]] | None = None,
) -> ImaginedRollout:
  """The rollout of `imagine` with the given sensor noise [B, hist_len + H, 93]
  (`imagination_noise`); the caller freezes `wm` and `critic` (`frozen`).

  `step` and `finish` are `imagine_step` and `finish_rollout` with these networks and
  `cfg.goal_anchor` (default) or compiled versions of them (`imagine_step_human` for the human rollout).
  Every step gets inputs of the same shapes and strides, so a compiled step is not
  recompiled per step (it is compiled twice: step 0's inputs need no grad).

  `extras(t)` are further positional arguments of step t, after the noise (default
  none); the human rollout passes `(point_mask, frame_mask, jitter[:, t : t + K],
  human_win[:, t : t + K])`."""
  horizon, k = cfg.horizon, GOAL_K
  b, hist_len = start.obs_hist.shape[:2]
  device = start.state.device
  kw = anchor_kwargs(cfg.goal_anchor)
  if step is None:
    step = functools.partial(imagine_step, wm, policy, obs_norm, goal_norm, **kw)
  if finish is None:
    finish = functools.partial(finish_rollout, wm, critic, goal_norm, **kw)
  window = reference_window(library, start.frame, horizon + k)
  num_rewarded = (library.remaining(start.frame) + 1).clamp(max=horizon)
  trunc_mask = torch.arange(horizon, device=device) < num_rewarded.unsqueeze(-1)
  bootstrap = None
  if cfg.clip_end_terminal:  # the clip's last transition lies within the horizon
    clip_ended = library.remaining(start.frame) + 1 <= horizon
    bootstrap = (~clip_ended).to(start.state.dtype)

  obs_hist = start.obs_hist + noise[:, :hist_len]  # noisy start history, oldest first
  prev_action = start.obs_hist[:, -1, PROPRIO_SLICES.prev_action].contiguous()
  state = start.state
  survival = torch.ones(b, device=device)
  steps: list[ImaginedStep] = []
  for t in range(horizon):
    out = step(
      state,
      obs_hist,
      prev_action,
      survival,
      window[:, t : t + k],
      start.ctx,
      noise[:, hist_len + t],
      *(extras(t) if extras is not None else ()),
    )
    steps.append(out)
    state, prev_action = out.next_state, out.action
    survival, obs_hist = out.survival, out.obs_hist
  return finish(
    start.state, steps, window[:, horizon : horizon + k], start.ctx, trunc_mask,
    bootstrap,
  )


def policy_objective(ro: ImaginedRollout, cfg: ImaginationCfg) -> Tensor:
  """Policy loss -J (Eq. 2 with truncation): the negative batch mean of the
  survival-weighted discounted rewards of the T_b rewarded steps plus
  γ^{T_b} ĉ_{T_b} V_{T_b}."""
  horizon = ro.rewards.shape[1]
  discount = cfg.gamma ** torch.arange(
    horizon + 1, device=ro.rewards.device, dtype=ro.rewards.dtype
  )
  weighted = discount[:horizon] * ro.survival[:, :horizon] * ro.rewards
  ret = torch.where(ro.trunc_mask, weighted, 0.0).sum(dim=1)
  tb = ro.num_rewarded.unsqueeze(-1)
  bootstrap = (
    discount[tb] * ro.survival.gather(1, tb) * ro.values.gather(1, tb)
  ).squeeze(-1)
  if ro.bootstrap is not None:
    bootstrap = bootstrap * ro.bootstrap
  return -(ret + bootstrap).mean()


def student_objective(
  ro: ImaginedRollout,
  cfg: ImaginationCfg,
  student: Literal["copy", "return", "direct"],
  action_weight: float,
  align_weight: float,
  scales: dict[str, Tensor] | None,
  recon_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
  """Human-student loss and its raw terms (detached). The terms: "return" = -J
  (`policy_objective`; the return and direct students), "copy" = the mean of
  `ro.action_err` and "align" = the mean of `ro.latent_err` over the rewarded steps
  (`trunc_mask`, max(#rewarded, 1) in the denominator; steps at t >= T_b are dropped
  with `torch.where`; copy and return students), "recon" = the mean of `ro.recon_err`
  likewise (the direct student only). The loss is the sum of weight x term / scale,
  with the weights 1, `action_weight`, `align_weight` and `recon_weight` and the
  `scales` (`TermScales.scales`; None divides by 1)."""
  count = ro.trunc_mask.sum().clamp(min=1)
  if student == "direct":
    terms = {
      "return": policy_objective(ro, cfg),
      "recon": torch.where(ro.trunc_mask, ro.recon_err, 0.0).sum() / count,
    }
  else:
    terms = {
      "copy": torch.where(ro.trunc_mask, ro.action_err, 0.0).sum() / count,
      "align": torch.where(ro.trunc_mask, ro.latent_err, 0.0).sum() / count,
    }
  if student == "return":
    terms = {"return": policy_objective(ro, cfg), **terms}
  weights = {
    "return": 1.0, "copy": action_weight, "align": align_weight, "recon": recon_weight
  }
  loss = sum(
    weights[name] * term / (1.0 if scales is None else scales[name])
    for name, term in terms.items()
  )
  return loss, {name: term.detach() for name, term in terms.items()}


class TermScales(nn.Module):
  """Running scale of each loss term `names`: an exponential average (`decay`) of the
  term's absolute value, started at the first value seen and floored at `floor`. A
  term not yet seen reads 1.0, so the first loss is the raw term. Buffers `scale` and
  `seen`, so the scales move with `.to` and save with `state_dict`."""

  def __init__(
    self, names: tuple[str, ...], decay: float = 0.99, floor: float = 1e-6
  ) -> None:
    super().__init__()
    self.names = names
    self.decay = decay
    self.floor = floor
    self.register_buffer("scale", torch.zeros(len(names)))
    self.register_buffer("seen", torch.zeros(len(names), dtype=torch.bool))

  def scales(self) -> dict[str, Tensor]:
    """The current scale of every term (detached)."""
    value = torch.where(self.seen, self.scale.clamp(min=self.floor), 1.0)
    return {name: value[i] for i, name in enumerate(self.names)}

  @torch.no_grad()
  def update(self, terms: dict[str, Tensor]) -> None:
    """Folds the absolute value of each of `terms` into its scale. A non-finite value
    is skipped (it would poison the scale, and so every later loss)."""
    for name, term in terms.items():
      i = self.names.index(name)
      size = term.detach().abs().to(self.scale.dtype)
      mixed = self.decay * self.scale[i] + (1.0 - self.decay) * size
      finite = torch.isfinite(size)
      self.scale[i] = torch.where(
        finite, torch.where(self.seen[i], mixed, size), self.scale[i]
      )
      self.seen[i] = self.seen[i] | finite


@torch.no_grad()
def gae_targets(ro: ImaginedRollout, cfg: ImaginationCfg) -> Tensor:
  """Critic targets R_t [B, H] (Eq. 3) on detached tensors: R_{T_b} = V_{T_b} and, for
  t < T_b, R_t = r̂_{t+1} + γ (1 - d̂_{t+1}) ((1 - λ) V_{t+1} + λ R_{t+1}), clamped
  to `cfg.target_max` when set. Entries at t >= T_b are not targets (finite filler)."""
  rewards, dones, values = ro.rewards.detach(), ro.dones.detach(), ro.values.detach()
  horizon = rewards.shape[1]
  tb = ro.num_rewarded
  targets = torch.empty_like(rewards)
  weight = ro.bootstrap.detach() if ro.bootstrap is not None else None
  ret = values[:, horizon]
  for t in reversed(range(horizon)):
    v_next = values[:, t + 1]
    if weight is not None:  # V_{T_b} counts with the bootstrap weight (0: clip ended)
      v_next = v_next * torch.where(t + 1 >= tb, weight, torch.ones_like(weight))
    next_ret = torch.where(t + 1 >= tb, v_next, ret)
    ret = rewards[:, t] + cfg.gamma * (1.0 - dones[:, t]) * (
      (1.0 - cfg.lam) * v_next + cfg.lam * next_ret
    )
    if cfg.target_max is not None:
      ret = ret.clamp(max=cfg.target_max)
    targets[:, t] = ret
  return targets


def target_values(critic: Critic, ro: ImaginedRollout) -> Tensor:
  """`critic`'s values V_0 .. V_H [B, H+1] of the rollout's states (detached inputs),
  for TD targets from another critic than the one that produced `ro.values`."""
  b, length = ro.local.shape[:2]
  return critic(
    ro.local.detach().flatten(0, 1),
    ro.goals.detach().flatten(0, 1),
    ro.ctx.detach().unsqueeze(1).expand(-1, length, -1).flatten(0, 1),
    ro.priv.detach().flatten(0, 1),
  ).view(b, length)


def critic_loss(
  critic: Critic, ro: ImaginedRollout, targets: Tensor, idx: Tensor
) -> Tensor:
  """Squared error of V(ŝ_t, g_t, xi, p_t) against `targets` on the entries `idx` of the
  flattened [B, H] grid (index b * H + t), averaged over the valid ones (t < T_b,
  `trunc_mask`): sum of err^2 over the valid entries / max(#valid, 1). Invalid entries
  are dropped with `torch.where`, not weighted by 0 (0 * inf = NaN). Inputs are
  detached. Fixed-size index sets and the mask avoid a data-dependent `nonzero` (host
  sync)."""
  horizon = targets.shape[1]
  b, t = idx // horizon, idx % horizon
  pred = critic(
    ro.local[b, t].detach(),
    ro.goals[b, t].detach(),
    ro.ctx[b].detach(),
    ro.priv[b, t].detach(),
  )
  mask = ro.trunc_mask[b, t]
  err2 = torch.where(mask, (pred - targets[b, t]).square(), 0.0)
  return err2.sum() / mask.to(pred.dtype).sum().clamp(min=1.0)


def fit_critic(
  critic: Critic,
  optimizer: torch.optim.Optimizer,
  ro: ImaginedRollout,
  cfg: ImaginationCfg,
  max_grad_norm: float = 1.0,
  targets: Tensor | None = None,
  loss_fn: Callable[[ImaginedRollout, Tensor, Tensor], Tensor] | None = None,
  average_grads: Callable[[], None] | None = None,
) -> tuple[Tensor, Tensor]:
  """Eq. 3 critic fit: fixed targets from `gae_targets`, `critic_epochs` passes over
  a shuffled permutation of all B * H entries in `critic_minibatches` minibatches,
  each with the mask-weighted loss of `critic_loss` (only t < T_b counts), global-norm
  clipping with the non-finite guard (`guarded_clip_grad_norm_`). Returns (mean
  minibatch loss, number of minibatch steps whose gradient was non-finite and zeroed),
  0-dim tensors. No host sync.

  `targets` (if already computed) and `loss_fn(ro, targets, idx)` (default
  `critic_loss` of `critic`) let the trainer pass compiled versions of the two;
  `average_grads()` (data parallel) runs after each backward, before the guard and the
  clipping."""
  if targets is None:
    targets = gae_targets(ro, cfg)
  if loss_fn is None:
    loss_fn = functools.partial(critic_loss, critic)
  num = targets.numel()
  losses, nonfinite = [], []
  for _ in range(cfg.critic_epochs):
    perm = torch.randperm(num, device=targets.device)
    for idx in perm.chunk(cfg.critic_minibatches):
      optimizer.zero_grad(set_to_none=True)
      loss = loss_fn(ro, targets, idx)
      loss.backward()
      if average_grads is not None:
        average_grads()
      nonfinite.append(guarded_clip_grad_norm_(critic.parameters(), max_grad_norm)[1])
      optimizer.step()
      losses.append(loss.detach())
  return torch.stack(losses).mean(), torch.stack(nonfinite).sum()


def guarded_clip_grad_norm_(
  params: Iterable[Tensor], max_norm: float
) -> tuple[Tensor, Tensor]:
  """`nn.utils.clip_grad_norm_` (global L2 norm clipped to `max_norm`, same
  coefficient) with a non-finite guard, on the device without a host sync.

  If the global norm is inf or NaN (any gradient entry is), plain clipping would
  multiply by 0 (inf norm) or NaN, leave NaN gradients and let the optimiser step
  write NaN into Adam's moments and the parameters for good. Instead every gradient
  is zeroed (non-finite entries -> 0, then all x 0), so the following optimiser step
  takes nothing from this batch (Adam's step then follows its decayed moments, which
  stay finite). Call it after any data-parallel gradient averaging, so every rank
  takes the same decision. Parameters without a gradient are skipped; the others'
  gradients share one device and dtype (one module).

  Returns (global norm before clipping, 1 if the gradient was non-finite else 0, as
  an int64 0-dim tensor), both on the gradients' device."""
  grads = [p.grad for p in params if p.grad is not None]
  if not grads:
    zero = torch.zeros(())
    return zero, zero.long()
  norm = nn.utils.get_total_norm(grads, 2.0)
  finite = torch.isfinite(norm)
  for g in grads:
    g.nan_to_num_(0.0, 0.0, 0.0)  # identity on finite entries
  coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)  # as clip_grad_norm_
  torch._foreach_mul_(grads, torch.where(finite, coef, 0.0))
  return norm, (~finite).long()


def clip_grads_(
  params: Iterable[Tensor], global_norm: float = 1.0, elementwise: float = 0.1
) -> tuple[Tensor, Tensor]:
  """Clips gradients in place: global norm to `global_norm` (with the non-finite
  guard of `guarded_clip_grad_norm_`), then each element to [-elementwise,
  elementwise]. Parameters without a gradient are skipped. Returns (global norm before
  clipping, non-finite flag) as `guarded_clip_grad_norm_`."""
  params = [p for p in params if p.grad is not None]
  norm, nonfinite = guarded_clip_grad_norm_(params, global_norm)
  nn.utils.clip_grad_value_(params, elementwise)
  return norm, nonfinite
