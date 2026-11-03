"""Train a SuperMimic tracking policy (Algorithm A.1).

Example (local smoke run):
  uv run python -m morphmimic.scripts.train \
    --subset-yaml $DATA/seed_g1_npz50/subsets/tiny.yaml --num-envs 512 \
    --iterations 60 --warmup-env-steps 20 --wm-batch 512 --imag-batch 512 \
    --log-every 10 --ckpt-every 50 --run-name smoke_tiny

Pick the GPU with --gpu. Resume a run
with --resume <run_dir>/latest.pt (same flags; --iterations may grow). A resume keeps
the checkpoint's --critic-target-clip setting (off for a checkpoint from before the
clamp) unless the command line names the flag. For the full set pass
--library-dtype float16 (halves the reference-motion memory).

Data parallel (paper Table A.5: gradients averaged across GPUs): --gpu-ids "[6,7]"
(or --gpu-ids 6 7; overrides --gpu) starts one process per GPU with torchrunx, each
simulating --num-envs envs; all three LRs are scaled by sqrt(#GPUs) (Table A.5; off
with --no-lr-scale-sqrt-world-size). Rank 0 prints the log lines (metrics pooled over
the ranks) and writes the run dir; every rank's output is also kept under
<run_dir>/torchrunx/. --resume loads the checkpoint on every rank.

Multi-robot (one robot per GPU): --robots g1 x2 t2
--subset-yamls <g1> <x2> <t2> --gpu-ids "[a,b,c]" --share wm|all|policy; rank r simulates
robots[r] on subset_yamls[r] (--robot and --subset-yaml are ignored), the world model
(--share wm), the world model, policy and critics (--share all), or the policy and
critics with a world model per robot (--share policy) is shared, every network gets the robot's
morphology vector, the LRs are not scaled. Each robot logs and checkpoints under
<run_dir>/<robot>/; rank 0 prints the pooled line and one line per robot. --resume
<run_dir> (or any robot's checkpoint file there) loads each robot's own checkpoint.

The learned-model phases run compiled as CUDA graphs (--compile reduce-overhead, the
default): the first iterations compile for a few minutes per batch-size configuration
(later runs load Inductor's on-disk cache); --compile none runs them eagerly.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.distributed as dist
import tyro
from mjlab.utils.torch import configure_torch_backends

from morphmimic.scripts.gpu_guard import select_gpus
from morphmimic.train.trainer import TrainCfg, Trainer

# Collective timeout of a data-parallel run: generous, since the ranks compile their
# CUDA graphs independently during the first iterations (minutes each).
PROCESS_GROUP_TIMEOUT = timedelta(minutes=30)


def _parse_gpu_ids(args: list[str]) -> list[int] | None:
  """`"[5,7]"`, `"5,7"` or `5 7` -> [5, 7]; `None` -> None."""
  text = " ".join(args).strip()
  if text == "None":
    return None
  inner = text.removeprefix("[").removesuffix("]")
  return [int(tok) for tok in inner.replace(",", " ").split()]


GpuIds = Annotated[
  list[int] | None,
  tyro.constructors.PrimitiveConstructorSpec(
    nargs="*",
    metavar="IDS",
    instance_from_str=_parse_gpu_ids,
    is_instance=lambda x: x is None or isinstance(x, list),
    str_from_instance=lambda ids: ["None"] if ids is None else [str(i) for i in ids],
  ),
]


@dataclass
class TrainArgs(TrainCfg):
  gpu: int = 0
  """Physical GPU index (exposed to the process as cuda:0)."""
  gpu_ids: GpuIds = None
  """Physical GPUs of a data-parallel run, one process each: "[6,7]" or 6 7. Overrides
  --gpu."""
  allow_any_gpu: bool = False
  """Ignored (every GPU is allowed); kept so old command lines still parse."""
  resume: str | None = None
  """Checkpoint to resume from (networks, normalisers, optimisers, schedule, reset
  sampler, RNG; not the replay, which warms up again); training continues from its
  iteration up to --iterations. Multi-robot runs: the run dir (each robot's
  latest.pt) or one robot's checkpoint file (each robot's file of that name)."""


def explicit_fields(argv: Sequence[str]) -> set[str]:
  """The `TrainCfg` fields the command line `argv` names (`--x`, `--no-x` of a
  boolean, `--x=value`; hyphens or underscores). A resume keeps the checkpoint's
  value of a `RESUME_KEPT_FIELDS` field unless the command line names it
  (`Trainer.load(explicit=)`)."""
  names = {f.name for f in fields(TrainCfg)}
  out = set()
  for token in argv:
    if not token.startswith("--"):
      continue
    name = token[2:].split("=", 1)[0].replace("-", "_")
    if name in names:
      out.add(name)
    elif name.startswith("no_") and name[3:] in names:
      out.add(name[3:])
  return out


def rank_resume_path(resume: str, robot: str) -> str:
  """Multi-robot resume: `robot`'s checkpoint for `--resume`. A robot's dir (it holds
  `latest.pt`, or is named `robot`) -> `<run>/<robot>/latest.pt`; a run dir ->
  `<resume>/<robot>/latest.pt`; a file -> the file of the same name in the run's
  `<robot>/` dir (`<run>/<any robot>/ckpt_N.pt` -> `<run>/<robot>/ckpt_N.pt`)."""
  path = Path(resume)
  if path.is_dir():
    if (path / "latest.pt").is_file() or path.name == robot:
      return str(path.parent / robot / "latest.pt")
    return str(path / robot / "latest.pt")
  return str(path.parent.parent / robot / path.name)


def run(
  cfg: TrainCfg,
  resume: str | None,
  distributed: bool = False,
  explicit: Collection[str] = (),
) -> None:
  """One training process: the only one, or (`distributed`) one torchrunx worker,
  which reads its rank from the env torchrunx sets, binds cuda:<local rank> (also for
  MuJoCo's EGL) and joins the NCCL process group. Rank r seeds with `cfg.seed + r`.
  `explicit`: the cfg fields the command line names (`explicit_fields`), passed to
  `Trainer.load` on resume."""
  rank = local_rank = 0
  if distributed:
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
      backend="nccl",
      device_id=torch.device("cuda", local_rank),
      timeout=PROCESS_GROUP_TIMEOUT,
    )
  try:
    configure_torch_backends()  # TF32 matmuls (paper: the world model runs at TF32).
    seed = cfg.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = Trainer(cfg, f"cuda:{local_rank}")
    try:
      if resume is not None:
        path = resume
        if trainer.multi:  # each robot resumes from its own checkpoint
          path = rank_resume_path(resume, trainer.cfg.robot)
        restored = trainer.load(path, explicit=explicit)
        if trainer.is_main:
          print(
            f"[morphmimic] resumed {path} at iteration {trainer.it}; restored"
            f" {', '.join(restored)}",
            flush=True,
          )
      if trainer.is_main:
        run_dir = trainer.run_dir.parent if trainer.multi else trainer.run_dir
        print(f"[morphmimic] run dir: {run_dir}", flush=True)
      if trainer.multi and trainer.is_main:
        robots = ", ".join(
          f"{robot} (GPU {gpu})"
          for robot, gpu in zip(cfg.robots or (), cfg.gpu_ids or [None], strict=False)
        )
        print(
          f"[morphmimic] multi-robot: {trainer.world_size} rank(s) x {cfg.num_envs} envs:"
          f" {robots}; shared {trainer.cfg.share}; lr x{trainer.lr_scale:.4f};"
          f" FiLM {trainer.cfg.decoder_film}; actor geometry"
          f" {trainer.cfg.actor_geometry}; latent align"
          f" {trainer.cfg.latent_align_weight:g}; pooled norms {trainer.cfg.pool_norms}",
          flush=True,
        )
      elif distributed and trainer.is_main:
        print(
          f"[morphmimic] data parallel: {trainer.world_size} ranks x {cfg.num_envs}"
          f" envs on GPUs {cfg.gpu_ids}; lr x{trainer.lr_scale:.4f}",
          flush=True,
        )
      trainer.train()
    finally:
      trainer.close()
  finally:
    if distributed:
      dist.destroy_process_group()


def launch(cfg: TrainCfg, resume: str | None, explicit: Collection[str] = ()) -> None:
  """Data parallel: one `run` process per GPU in `cfg.gpu_ids` via torchrunx (as mjlab
  launches multi-GPU training). The run name is fixed here so all ranks share one run
  dir; rank 0's output is relayed to this process's stderr, every rank's is kept under
  `$TORCHRUNX_LOG_DIR` (default <run_dir>/torchrunx)."""
  import torchrunx

  # Importing this module by name lets torchrunx pickle `run` by reference.
  from morphmimic.scripts import train as entry  # noqa: PLW0406

  assert cfg.gpu_ids is not None
  if not cfg.run_name:
    cfg = replace(cfg, run_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
  run_dir = Path(cfg.log_dir) / cfg.run_name
  os.environ.setdefault("TORCHRUNX_LOG_DIR", str(run_dir / "torchrunx"))
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  print(
    f"[morphmimic] launching {len(cfg.gpu_ids)} processes on GPUs {cfg.gpu_ids}"
    f" (logs: {os.environ['TORCHRUNX_LOG_DIR']})",
    flush=True,
  )
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=len(cfg.gpu_ids),
    backend=None,  # `run` creates the NCCL group itself.
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*", "WANDB*"),
  ).run(entry.run, cfg, resume, True, frozenset(explicit))


def main() -> None:
  args = tyro.cli(TrainArgs)
  gpus = args.gpu_ids if args.gpu_ids is not None else [args.gpu]
  select_gpus(gpus, args.allow_any_gpu)
  cfg = TrainCfg(**{f.name: getattr(args, f.name) for f in fields(TrainCfg)})
  explicit = explicit_fields(sys.argv[1:])
  if len(gpus) > 1:
    launch(cfg, args.resume, explicit)
  else:
    run(cfg, args.resume, explicit=explicit)


if __name__ == "__main__":
  main()
