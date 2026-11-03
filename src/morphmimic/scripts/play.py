"""Watch a checkpoint track one clip in mjlab's viewer, with the reference as a ghost.

Example:
  uv run python -m morphmimic.scripts.play --ckpt logs/smoke_tiny/latest.pt \
    --subset-yaml $DATA/seed_g1_npz50/subsets/tiny.yaml --clip 0

A checkpoint trained with `--kp-interface` can also play through the human skeleton
goal: add `--interface human --human-points {full,five,six,three} --kp-horizon 1..10`
(fixed masks, as in `scripts/eval.py`; `--human-root` if the sidecars are not under
`<clip root>/human50`; `--no-human-heading` without the relative orientation). Clips
without human sidecars are refused for it (ValueError), as in the benchmark.

One env in play + eval mode (no DR, no observation or reset noise, floor friction 1.0)
plays `--clip` (name or index in the subset) from its first frame with the
deterministic policy, and restarts it whenever it fails or reaches the clip's last frame.
`--viewer auto` uses the native MuJoCo viewer when a display is available, otherwise
viser (open the printed URL in a browser).

Limitations: the whole subset's motion library is loaded although one clip plays (use
a small subset for long datasets); the viewer's checkpoint hot-swap is not wired up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from torch import Tensor

from morphmimic.constants import GOAL_DIM, HUMAN_ENV_DIM, PROPRIO_DIM, R_MAX
from morphmimic.data.manifest import load_subset
from morphmimic.env.commands import MultiMotionCommand
from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.eval.benchmark import require_human_sidecars
from morphmimic.eval.policy_loader import (
  InterfaceArgs,
  PolicyFn,
  goal_anchor_kwargs,
  load_policy,
)
from morphmimic.scripts.gpu_guard import select_gpu
from morphmimic.train.collector import GOAL_GROUPS
from morphmimic.train.obs_history import ObsHistory


@dataclass
class PlayArgs(InterfaceArgs):
  ckpt: str
  """Training checkpoint (`ckpt_<it>.pt` or `latest.pt`)."""
  subset_yaml: str
  """Subset that contains the clip."""
  clip: str = "0"
  """Clip name (npz stem, as in the manifest) or index in the subset."""
  gpu: int = 0
  """Physical GPU index (exposed to the process as cuda:0)."""
  allow_any_gpu: bool = False
  """Ignored (every GPU is allowed); kept so old command lines still parse."""
  viewer: Literal["auto", "native", "viser"] = "auto"
  num_steps: int | None = None
  """Stop after this many env steps (default: run until the viewer is closed)."""
  robot: str | None = None
  """Robot to play on (default: the checkpoint's robot)."""


def resolve_clip(subset_yaml: str, clip: str) -> int:
  """Index of `clip` (a clip name or an index) in the subset."""
  names = [Path(p).stem for p in load_subset(subset_yaml)]
  if clip in names:
    return names.index(clip)
  if clip.lstrip("-").isdigit() and -len(names) <= int(clip) < len(names):
    return int(clip) % len(names)
  raise SystemExit(f"clip {clip!r} is neither a name nor an index of {subset_yaml}")


class ClipPlayback:
  """mjlab viewer env (`mjlab.viewer.EnvProtocol`) in which every env of `env` plays
  clip `clip_id` from its first frame and restarts it when it ends or fails.

  `MorphMimicEnv` runs with `auto_reset=False` and the policy's obs history lives
  outside the env, so this wrapper resets finished envs, keeps the `ObsHistory` of the
  `policy` obs and the goals (`GOAL_GROUPS`), and serves both as the viewer's
  observation: `get_observations()` -> (obs history flat, goals). Each finished
  playthrough prints whether it failed and its normalised return (benchmark definition).
  """

  def __init__(self, env: MorphMimicEnv, clip_id: int, hist_len: int) -> None:
    self._env = env
    command = env.command_manager.get_term(COMMAND_NAME)
    assert isinstance(command, MultiMotionCommand)
    self.command = command
    self.clip_id = clip_id
    self.clip_name = command.library.clip_names[clip_id]
    self.clip_len = int(command.library.clip_len[clip_id])
    self.num_envs = env.num_envs
    self.history = ObsHistory(env.num_envs, hist_len, PROPRIO_DIM, env.device)
    self.goals = {
      "goal": torch.zeros(env.num_envs, GOAL_DIM, device=env.device),
      "goal_kp": torch.zeros(env.num_envs, HUMAN_ENV_DIM, device=env.device),
    }
    self.ep_return = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    self.playthroughs: list[dict[str, Any]] = []
    self.reset()

  @property
  def unwrapped(self) -> MorphMimicEnv:
    return self._env

  @property
  def device(self) -> str:
    return self._env.device

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self._env.cfg

  def get_observations(self) -> tuple[Tensor, dict[str, Tensor]]:
    return self.history.flat(), self.goals

  def reset(self) -> tuple[tuple[Tensor, dict[str, Tensor]], dict]:
    self._restart(torch.arange(self.num_envs, device=self.device))
    return self.get_observations(), {}

  def step(self, action: Tensor):
    frame = self.command.global_frame.clone()
    obs, reward, terminated, truncated, extras = self._env.step(action)
    self.ep_return += reward.double()
    self.history.push(obs["policy"])
    self.goals = {g: obs[g].clone() for g in GOAL_GROUPS}
    finished = (terminated | truncated).nonzero().squeeze(-1)
    if finished.numel() > 0:
      start = int(self.command.library.clip_start[self.clip_id])
      for i in finished.tolist():
        failed = bool(terminated[i])
        result = {
          "env": i,
          "failed": failed,
          "frame": int(frame[i]) - start,
          "norm_return": float(self.ep_return[i]) / (R_MAX * (self.clip_len - 1)),
        }
        self.playthroughs.append(result)
        what = f"failed at frame {result['frame']}" if failed else "completed"
        print(
          f"[morphmimic] {self.clip_name}: {what} of {self.clip_len}, normalised "
          f"return {result['norm_return']:.3f}",
          flush=True,
        )
      self._restart(finished)
    return obs, reward, terminated, truncated, extras

  def close(self) -> None:
    self._env.close()

  def _restart(self, env_ids: Tensor) -> None:
    self.command.assign_clips(env_ids, torch.full_like(env_ids, self.clip_id))
    obs, _ = self._env.reset(env_ids=env_ids)
    self.history.reset(env_ids, obs["policy"][env_ids])
    for g in GOAL_GROUPS:
      self.goals[g][env_ids] = obs[g][env_ids]
    self.ep_return[env_ids] = 0.0


def viewer_policy(policy_fn: PolicyFn):
  """The viewer passes `get_observations()` to the policy."""

  def policy(obs: tuple[Tensor, dict[str, Tensor]]) -> Tensor:
    return policy_fn(*obs)

  return policy


def main() -> None:
  args = tyro.cli(PlayArgs)
  select_gpu(args.gpu, args.allow_any_gpu)
  configure_torch_backends()
  device = "cuda:0"
  clip_id = resolve_clip(args.subset_yaml, args.clip)
  policy_fn, meta = load_policy(
    args.ckpt,
    device,
    interface=args.interface,
    human_points=args.human_points,
    kp_horizon=args.kp_horizon,
    human_heading=args.human_heading,
    human_drift=args.human_drift,
    **goal_anchor_kwargs(args.goal_anchor),
  )
  cfg = morphmimic_env_cfg(
    args.robot or meta.robot,
    subset_yaml=args.subset_yaml,
    num_envs=1,
    play=True,
    eval=True,
    human_root=args.human_root,
    goal_anchor=args.goal_anchor,
  )
  sim = MorphMimicEnv(cfg, device=device)
  if args.interface == "human":
    try:
      require_human_sidecars(sim, args.subset_yaml, args.human_root)
    except ValueError:
      sim.close()
      raise
  env = ClipPlayback(sim, clip_id, meta.hist_len)
  print(
    f"[morphmimic] playing {env.clip_name} ({env.clip_len} frames) with {args.ckpt} "
    f"(iteration {meta.iteration}, {meta.mode})",
    flush=True,
  )

  viewer = args.viewer
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  viewer_cls = NativeMujocoViewer if viewer == "native" else ViserPlayViewer
  try:
    viewer_cls(env, viewer_policy(policy_fn)).run(num_steps=args.num_steps)
  finally:
    env.close()


if __name__ == "__main__":
  main()
