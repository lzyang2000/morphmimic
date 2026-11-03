"""Deterministic policy from a training checkpoint, shared by the benchmark
and the playback script.

The checkpoint is the dict of `Trainer.checkpoint()`: the policy is rebuilt from its
`cfg` (`policy_hidden`, `act`, `hist_len`, as `Trainer` builds it) and, when it was
trained with the morphology input, its `morph` entry (the actor morphology vector's
width and FiLM); the decoder's `morph_actor` buffer loads with the state dict and is
appended to every input. It acts through the same input path as collection and
imagination (`imagination.act`: running normalisers on the stacked obs and the goal),
without action noise. The policy function is given every goal group (as the collector
gives them) and acts on one of them, by `interface`: "motion" (default) on the motion
goal, `goals["goal"]`; "human" through the human encoder on the raw human goal,
`goals["goal_kp"]` (the 24 SMPL joints of the operator and the robot torso
quaternion, keypoint goal version 10), under a fixed mask (the
`human_points` slots, the first `kp_horizon` frames, the relative orientation on or
off). A human checkpoint (its
policy has `human_encoder.*` parameters) also runs the motion interface, unless it is a
direct human policy (`policy_goal="human"`, v9): its motion encoder was never trained,
so it runs the human interface only.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from morphmimic.constants import (
  DT,
  GOAL_DIM,
  GOAL_K,
  GOAL_VERSION,
  HUMAN_FEAT_DIM,
  HUMAN_HEADING_DIM,
  PROPRIO_DIM,
  GoalAnchor,
)
from morphmimic.model.human import (
  HumanPoints,
  drift_human_goal,
  fixed_human_masks,
  sample_human_drift,
)
from morphmimic.model.policy import Policy, PolicyCfg
from morphmimic.robots import check_imu_body
from morphmimic.robots.morphology import morphology_vector
from morphmimic.train.imagination import act, act_human
from morphmimic.train.normalizers import RunningNormalizer
from morphmimic.train.trainer import (
  _stale_kp_error,
  _stale_kp_parts,
  _stale_kp_warning,
  _without_stale_kp_parts,
  check_goal_anchor,
)

Interface = Literal["motion", "human"]
"""The goal interface a policy is evaluated through."""


@dataclass(kw_only=True)
class InterfaceArgs:
  """Goal-interface flags shared by the evaluation, playback, render and drift scripts:
  `--interface`, `--human-points`, `--kp-horizon`,
  `--human-heading`, `--goal-anchor` (arguments of `load_policy`; the anchor also goes
  to the env) and `--human-root` (where the env finds the human sidecars).
  Keyword-only, so a script's argument dataclass inherits them ahead of its own
  required fields."""

  interface: Interface = "motion"
  """Goal interface the policy acts through: the motion goal, or the human skeleton
  goal (needs a `kp_interface` checkpoint)."""
  human_points: HumanPoints = "full"
  """Human interface: the tracked slots, all 24 (`full`), `five` (pelvis, ankles,
  wrists), `six` (five plus the chest) or `three` (head and wrists). Ignored with
  `--interface motion`."""
  kp_horizon: int = GOAL_K
  """Human interface: visible preview frames, 1..10. Ignored with
  `--interface motion`."""
  human_heading: bool = True
  """Human interface: give the policy the relative orientation (the human anchor's
  orientation relative to the robot pelvis, per future frame); `--no-human-heading`
  zeroes it after its normaliser, as a masked row (training sees that only with
  `--kp-heading-frac` below 1). Ignored with `--interface motion`."""
  human_drift: bool = False
  """Human interface: move the human goal by a slow drift plus per-frame noise
  (`HUMAN_DRIFT`, training's `--kp-drift`), one draw per env, to measure robustness
  to the headset's wandering. Ignored with `--interface motion`."""
  human_root: str | None = None
  """Root of the clips' human sidecars (default `<clip root>/human50`)."""
  goal_anchor: GoalAnchor = "reference"
  """Frame of the motion goal's anchor position; must be the checkpoint's
  (`--goal-anchor` of its training run)."""


def goal_anchor_kwargs(goal_anchor: GoalAnchor) -> dict[str, GoalAnchor]:
  """`goal_anchor=` for `load_policy`: empty for the default "reference", so a
  default call is exactly the one of before the switch."""
  return {} if goal_anchor == "reference" else {"goal_anchor": goal_anchor}


_HUMAN_ENCODER_PREFIX = "human_encoder."
PolicyFn = Callable[[Tensor, dict[str, Tensor]], Tensor]
"""(obs history [B, hist_len * 93] as `ObsHistory.flat`, goals {"goal": [B, 670],
"goal_kp": [B, 1684]}) -> action."""


@dataclass(frozen=True)
class PolicyMeta:
  path: str
  iteration: int
  hist_len: int
  """Length of the obs history the policy expects (the caller stacks it)."""
  train_cfg: dict[str, Any]
  """The `TrainCfg` of the run, as saved (`dataclasses.asdict`)."""
  robot: str
  """Robot the policy was trained on (`TrainCfg.robot`; "g1" for checkpoints from
  before the field)."""
  interface: str
  """"motion" or "human"."""
  mode: str
  """"motion", or "human-<points>-h<horizon>" (e.g. "human-five-h1"), plus "-noyaw"
  without the relative orientation, for the human
  interface: names the evaluation in result files."""


def mixed_morphology_vector(
  a: str, b: str, alpha: float, actor_geometry: bool = True
) -> torch.Tensor:
  """`(1 - alpha) * vec(a) + alpha * vec(b)` over the actor morphology vectors of two
  registry robots: a point on the segment between two training bodies."""
  va = morphology_vector(a, actor=True, actor_geometry=actor_geometry)
  vb = morphology_vector(b, actor=True, actor_geometry=actor_geometry)
  return (1.0 - float(alpha)) * va + float(alpha) * vb


def load_policy(
  ckpt_path: str | Path,
  device: str,
  interface: Interface = "motion",
  human_points: HumanPoints = "full",
  kp_horizon: int = GOAL_K,
  morph_robot: str | None = None,
  morph_mix: tuple[str, str, float] | None = None,
  morph_vector: torch.Tensor | None = None,
  norm_from: str | Path | None = None,
  goal_anchor: GoalAnchor = "reference",
  human_heading: bool = True,
  human_drift: bool = False,
) -> tuple[PolicyFn, PolicyMeta]:
  """Policy and normalisers of a checkpoint on `device`, as a no-grad callable.

  Loaded on the CPU first, then moved (as `Trainer.load`). The normalisers are in eval
  mode, so they never update. A checkpoint with a `morph` entry (morphology input)
  rebuilds the decoder with the actor vector's width and FiLM; its robot's vector,
  the `decoder.morph_actor` buffer of the state dict, goes with every action. The
  policy has the human encoder iff the checkpoint does.

  `interface="motion"` acts on `goals["goal"]`; `human_points`, `kp_horizon` and
  `human_heading` are ignored. `interface="human"` acts on the raw human goal
  `goals["goal_kp"]` [B, 1684] through the human encoder (`act_human`) with the fixed
  masks of `fixed_human_masks(batch, human_points, kp_horizon, heading=human_heading)`
  (built for the first batch size seen, again whenever it changes); the mode gets
  "-noyaw" when `human_heading` is False. `human_drift` moves the human goal by
  `drift_human_goal`, one `sample_human_drift` draw per env (drawn with the masks) at
  the env's step count since then; the mode gets "-drift". It needs a checkpoint of a `kp_interface` run
  (human encoder, `human_norm` and `heading_norm`), else raises ValueError. A
  checkpoint of a `policy_goal="human"` run (the direct human policy) runs the human
  interface only: `interface="motion"` raises ValueError. A mask
  setting out of range also raises ValueError, before the checkpoint is read, and so
  does a checkpoint whose goal inputs are another version (`GOAL_VERSION`; none saved = 1)
  or whose robot read its IMU on another body (`check_imu_body`), or that was trained
  with another `goal_anchor` (`check_goal_anchor`; a cfg without one: "reference"),
  whose motion goal the caller's env must build.

  A checkpoint with keypoint parts of an older keypoint goal version (the five-point
  interface of versions 1-2, the human interface of versions 3-9; as `Trainer.load`) is
  refused by the human interface (ValueError); the motion interface drops those
  parts with a warning.

  `morph_robot` replaces the stored actor morphology vector with that robot's
  (`morphology_vector(..., actor=True)`, transfer); a checkpoint without
  the morphology input then raises ValueError. `morph_mix=(a, b, alpha)` replaces it
  with `(1 - alpha) * vec(a) + alpha * vec(b)` (`mixed_morphology_vector`);
  `morph_vector` replaces it with the given actor vector itself (a
  fitted one), which must have the stored vector's width.
  Alpha outside [0, 1], a wrong width, or two overrides at once raise ValueError.
  """
  if sum(x is not None for x in (morph_robot, morph_mix, morph_vector)) > 1:
    raise ValueError("pass one of morph_robot, morph_mix and morph_vector")
  if morph_mix is not None and not 0.0 <= float(morph_mix[2]) <= 1.0:
    raise ValueError(f"morph_mix alpha must be in [0, 1], got {morph_mix[2]!r}")
  if interface not in ("motion", "human"):
    raise ValueError(f"Expected interface 'motion' or 'human', got {interface!r}.")
  if interface == "human":
    fixed_human_masks(1, human_points, kp_horizon, torch.device("cpu"))
  ckpt = torch.load(ckpt_path, map_location="cpu")
  goal_version = int(ckpt.get("goal_version", 1))
  if goal_version != GOAL_VERSION:
    raise ValueError(
      f"checkpoint {ckpt_path} expects goal inputs version {goal_version}; this code"
      f" builds version {GOAL_VERSION} (position-free). Load it"
      " with the code it was trained with."
    )
  check_imu_body(ckpt, ckpt_path)
  cfg = ckpt["cfg"]
  check_goal_anchor(cfg, goal_anchor, ckpt_path)
  missing = [k for k in ("policy_hidden", "act", "hist_len") if k not in cfg]
  if missing:
    raise ValueError(f"checkpoint {ckpt_path}: cfg has no {missing}")
  human_policy = cfg.get("policy_goal", "motion") == "human"
  if human_policy and interface == "motion":
    raise ValueError(
      f"checkpoint {ckpt_path}: this checkpoint's policy reads the human goal"
      " (--policy-goal human; its motion encoder was never trained); use --interface"
      " human"
    )
  morph = ckpt.get("morph")
  stale = _stale_kp_parts(ckpt)
  if stale is not None:
    if interface == "human":
      raise ValueError(
        _stale_kp_error(ckpt_path, ckpt, stale, "with --interface motion")
      )
    warnings.warn(_stale_kp_warning(ckpt_path, ckpt, stale), stacklevel=2)
    ckpt = _without_stale_kp_parts(ckpt, stale)
  has_human = any(k.startswith(_HUMAN_ENCODER_PREFIX) for k in ckpt["policy"])
  if interface == "human" and not (
    has_human and "human_norm" in ckpt and "heading_norm" in ckpt
  ):
    raise ValueError(
      f"checkpoint has no human encoder (or no human_norm, heading_norm): {ckpt_path}"
    )
  policy = Policy(
    PolicyCfg(
      enc_hidden=tuple(cfg["policy_hidden"]),
      dec_hidden=tuple(cfg["policy_hidden"]),
      act=cfg["act"],
      hist_len=int(cfg["hist_len"]),
      latent=int(cfg.get("latent", 32)),
      morph_dim=0 if morph is None else int(morph["actor"].numel()),
      film=False if morph is None else bool(morph["film"]),
      human=has_human,
      recon=human_policy,
    )
  )
  policy.load_state_dict(ckpt["policy"])
  if morph_robot is not None or morph_mix is not None or morph_vector is not None:
    if morph is None:
      raise ValueError(f"checkpoint {ckpt_path} has no morphology input")
    geometry = bool(cfg.get("actor_geometry", True))
    if morph_vector is not None:
      vec = torch.as_tensor(morph_vector, dtype=torch.float32).reshape(-1)
      if vec.numel() != morph["actor"].numel():
        raise ValueError(
          f"morph_vector has {vec.numel()} values; the checkpoint's actor morphology"
          f" vector has {morph['actor'].numel()}"
        )
    elif morph_mix is not None:
      vec = mixed_morphology_vector(*morph_mix, actor_geometry=geometry)
    else:
      vec = morphology_vector(morph_robot, actor=True, actor_geometry=geometry)
    policy.set_morphology(vec)
  # `norm_from`: another checkpoint's observation and goal normalisers (same layouts,
  # e.g. another rank of a multi-robot run), the weights staying this checkpoint's
  # (the per-robot normaliser as a transfer candidate).
  norms = ckpt
  if norm_from is not None:
    norms = torch.load(norm_from, map_location="cpu", weights_only=False)
    for key in ("obs_norm", "goal_norm"):
      if key not in norms:
        raise ValueError(f"norm_from checkpoint {norm_from} has no {key}")
  obs_norm = RunningNormalizer(PROPRIO_DIM)
  obs_norm.load_state_dict(norms["obs_norm"])
  goal_norm = RunningNormalizer(GOAL_DIM)
  goal_norm.load_state_dict(norms["goal_norm"])
  policy = policy.to(device).eval()
  obs_norm = obs_norm.to(device).eval()
  goal_norm = goal_norm.to(device).eval()

  @torch.no_grad()
  def motion_fn(obs_hist_flat: Tensor, goals: dict[str, Tensor]) -> Tensor:
    return act(policy, obs_norm, goal_norm, obs_hist_flat, goals["goal"])

  policy_fn = motion_fn
  mode = "motion"
  if interface == "human":
    human_norm = RunningNormalizer(HUMAN_FEAT_DIM)
    human_norm.load_state_dict(ckpt["human_norm"])
    human_norm = human_norm.to(device).eval()
    heading_norm = RunningNormalizer(HUMAN_HEADING_DIM)
    heading_norm.load_state_dict(ckpt["heading_norm"])
    heading_norm = heading_norm.to(device).eval()
    masks: tuple[Tensor, Tensor, Tensor] | None = None
    drift = step = torch.zeros(0)

    @torch.no_grad()
    def human_fn(obs_hist_flat: Tensor, goals: dict[str, Tensor]) -> Tensor:
      nonlocal masks, drift, step
      batch = obs_hist_flat.shape[0]
      dev = obs_hist_flat.device
      if masks is None or masks[0].shape[0] != batch:
        masks = fixed_human_masks(
          batch, human_points, kp_horizon, dev, heading=human_heading
        )
        drift = sample_human_drift(batch, dev)
        step = torch.zeros(batch, device=dev)
      goal = goals["goal_kp"]
      if human_drift:
        frames = step[:, None] + torch.arange(GOAL_K, device=dev)
        goal = drift_human_goal(goal, drift, frames * DT)
        step += 1
      return act_human(
        policy, obs_norm, human_norm, heading_norm, obs_hist_flat, goal, *masks
      )

    policy_fn = human_fn
    mode = (
      f"human-{human_points}-h{kp_horizon}"
      + ("" if human_heading else "-noyaw")
      + ("-drift" if human_drift else "")
    )

  meta = PolicyMeta(
    path=str(ckpt_path),
    iteration=int(ckpt["iteration"]),
    hist_len=int(cfg["hist_len"]),
    train_cfg=dict(cfg),
    robot=cfg.get("robot", "g1"),
    interface=interface,
    mode=mode,
  )
  return policy_fn, meta
