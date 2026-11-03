"""Render clips played by a checkpoint to mp4 (offscreen, ghost reference overlaid).

    uv run python -m morphmimic.scripts.render --ckpt logs/run/ckpt_8000.pt \
      --subset-yaml $DATA/seed_g1_npz50/subsets/1h.yaml \
      --clips jump_ff_360_R_002__A296_M turn_jump_360_003__A051 --out-dir videos

With `--interface human --human-points {full,five,six,three} --kp-horizon 1..10` a
checkpoint trained with `--kp-interface` plays through the human skeleton goal (fixed
masks, as in `scripts/eval.py`; `--human-root` as there); its videos are named
`<clip>_human-<points>-h<horizon>.mp4` (`-noyaw` added with `--no-human-heading`).
`--kp-trace` draws that command at the env origin (per enabled slot of the clip's
human window: a sphere at the current target, a line through the preview frames, a
fading trail of past targets; `morphmimic.eval.kp_trace`), and `--no-ghost` hides the
reference robot, which the human policy never sees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from morphmimic.eval.policy_loader import InterfaceArgs
from morphmimic.scripts.gpu_guard import select_gpu


@dataclass
class RenderArgs(InterfaceArgs):
  ckpt: str
  subset_yaml: str
  clips: list[str] = field(default_factory=lambda: ["0"])
  """Clip names (npz stems) or indices in the subset."""
  out_dir: str = "videos"
  gpu: int = 0
  allow_any_gpu: bool = False
  width: int = 640
  height: int = 480
  fps: int = 50
  """Output frame rate; 50 = real time (one control step per frame)."""
  max_seconds: float = 20.0
  """Stop a clip after this many seconds even if it has not ended."""
  robot: str | None = None
  """Robot to render (default: the checkpoint's robot)."""
  morph_robot: str | None = None
  """Checkpoint with the morphology input: act with this robot's actor morphology
  vector instead of the stored one (as `scripts/eval.py --morph-robot`)."""
  kp_trace: bool = False
  """Draw the human command as traces (human interface only)."""
  ghost: bool = True
  """Draw the reference robot as a ghost."""
  trail_seconds: float = 1.0
  """Length of the trace's trail of past targets."""
  camera_distance: float = 5.0
  camera_elevation: float = -45.0
  camera_azimuth: float = 90.0


def video_path(out_dir: Path, clip_name: str, mode: str) -> Path:
  """`<out_dir>/<clip>.mp4` for the motion interface, `<clip>_<mode>.mp4` otherwise
  (mode as `PolicyMeta.mode`, e.g. `human-five-h3`)."""
  suffix = "" if mode == "motion" else f"_{mode}"
  return out_dir / f"{clip_name}{suffix}.mp4"


def main() -> None:
  args = tyro.cli(RenderArgs)
  if args.kp_trace and args.interface != "human":
    raise ValueError("--kp-trace draws the human command: use --interface human.")
  select_gpu(args.gpu, args.allow_any_gpu)
  os.environ.setdefault("MUJOCO_GL", "egl")

  import mediapy as media
  import torch
  from mjlab.utils.torch import configure_torch_backends

  from morphmimic.constants import DT
  from morphmimic.env.env_cfg import morphmimic_env_cfg
  from morphmimic.env.morphmimic_env import MorphMimicEnv
  from morphmimic.eval.kp_trace import KeypointTrace
  from morphmimic.eval.policy_loader import goal_anchor_kwargs, load_policy
  from morphmimic.model.human import fixed_points
  from morphmimic.scripts.play import ClipPlayback, resolve_clip

  configure_torch_backends()
  device = "cuda:0"
  policy_fn, meta = load_policy(
    args.ckpt,
    device,
    interface=args.interface,
    human_points=args.human_points,
    kp_horizon=args.kp_horizon,
    human_heading=args.human_heading,
    human_drift=args.human_drift,
    **goal_anchor_kwargs(args.goal_anchor),
    morph_robot=args.morph_robot,
  )
  cfg = morphmimic_env_cfg(
    args.robot or meta.robot,
    subset_yaml=args.subset_yaml,
    num_envs=1,
    eval=True,
    human_root=args.human_root,
    goal_anchor=args.goal_anchor,
  )
  cfg.viewer.width = args.width
  cfg.viewer.height = args.height
  cfg.viewer.distance = args.camera_distance
  cfg.viewer.elevation = args.camera_elevation
  cfg.viewer.azimuth = args.camera_azimuth
  env = MorphMimicEnv(cfg, device, render_mode="rgb_array")
  point_mask = fixed_points(1, args.human_points, torch.device("cpu"))[0]
  overlay: dict[str, object] = {"trace": None, "cmd": None}
  draw_managers = env.update_visualizers

  def visualize(visualizer) -> None:
    """Debug-vis callback of `env.render()`: the managers' overlays (the ghost) unless
    `--no-ghost`, then the human command traces (the window's frame 0 as the current
    target, the window as the preview, both at the env origin)."""
    if args.ghost:
      draw_managers(visualizer)
    trace, cmd = overlay["trace"], overlay["cmd"]
    if trace is None or cmd is None:
      return
    window = cmd.human_future()[0, :, :, :3] + env.scene.env_origins[0]
    future = window.cpu().numpy()
    trace.draw(visualizer, future[0], future)

  env.update_visualizers = visualize  # type: ignore[method-assign]
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  max_steps = int(args.max_seconds / DT)
  try:
    for clip in args.clips:
      clip_id = resolve_clip(args.subset_yaml, clip)
      pb = ClipPlayback(env, clip_id, meta.hist_len)
      overlay["cmd"] = pb.command
      overlay["trace"] = (
        KeypointTrace(int(args.trail_seconds / DT), args.kp_horizon, point_mask.tolist())
        if args.kp_trace
        else None
      )
      frames = [env.render()]
      with torch.no_grad():
        for _ in range(max_steps):
          hist, goals = pb.get_observations()
          pb.step(policy_fn(hist, goals))
          frames.append(env.render())
          if pb.playthroughs:
            break
      result = pb.playthroughs[0] if pb.playthroughs else {"failed": None}
      path = video_path(out_dir, pb.clip_name, meta.mode)
      media.write_video(str(path), frames, fps=args.fps)
      print(f"[morphmimic] wrote {path} ({len(frames)} frames, {len(frames) * DT:.1f} s) result={result}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
