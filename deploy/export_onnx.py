"""One ONNX policy for every robot of a shared-policy run (or for one robot's
checkpoint), with each robot's inputs in a json file and a manifest that shows the
policy is shared.

Given one checkpoint (a single-robot run, or one robot of any run), it exports that
robot's policy with the same files.

A multi-robot run writes one checkpoint per robot (`<run>/<robot>/ckpt_<it>.pt`), even
when the policy is shared (`--share all` or `--share policy`). The export first checks
that the shared networks are bit-identical across those checkpoints, then writes:

- `policy.onnx`: the shared policy's motion interface (`action_motion`) and, for a
  checkpoint with a human encoder, its human interface (`action_human`). The per-robot
  parts are inputs, not constants: the robot's input-normaliser statistics and its
  morphology vector. One file serves every robot.
- `<robot>.json`: those per-robot inputs, read from the robot's checkpoint, and the
  robot's constants (`robot`: joint map, default pose, action scale, kinematic tree,
  timing; baked in `deploy/sim2sim/robots/<robot>.json`), so the folder holds all the
  robot side needs. `goal_anchor` names the motion goal's anchor frame the policy
  was trained with (`--goal-anchor`, which must match the checkpoints': "reference",
  or "robot", which needs the robot's pose at deploy); `human_interface` whether the
  ONNX has the human interface; `policy_goal` the goal the policy reads ("motion", or
  "human" for a direct human policy: no `action_motion`).
- `MANIFEST.txt`: the shared-network check (a fingerprint per network and robot), the
  per-robot entries, the ONNX inputs and outputs, a parity check against the PyTorch
  policy, and the SHA-256 of every file.
- `SHA256SUMS`: the same digests, for `sha256sum -c SHA256SUMS`.

A checkpoint with a human encoder (keypoint goal version 10: `human_encoder.*`,
`human_norm` and `heading_norm`) also exports `action_human`: the graph takes the
human features (`human_features_from_goal`, built outside it from the raw human poses),
the slot and frame masks, the raw heading block (`heading_features`: the human Spine3's
orientation relative to the robot torso, rot6d per future frame) and its row mask
(False when unknown), and applies `human_norm` and `heading_norm`, `mask_human_goal`
and the human encoder, as `imagination.act_human` does after the feature build. The
sim2sim node feeds it under `--interface human` (PICO on the uniform skeleton, no GMR). A checkpoint's
five-point keypoint parts (keypoint goal versions 1-2) are dropped with a warning, as
`Trainer.load` drops them; its five-point deploy path is in this
repository's earlier release (commit aa40fe4). So are the human parts of a version 3-6 checkpoint (human
orientations relative to the robot pelvis, no heading error, the yaw-only heading
error, or the heading block fed raw): it exports its motion interface only.

A checkpoint of a `--policy-goal human` run (the direct human policy, v9: the human
encoder is the policy's encoder; its motion encoder was never trained) exports
`action_human` only: no `action_motion`, and no `goal_motion`, `goal_mean`, `goal_var`
inputs. Its reconstruction head (`recon_head.*`, training only) is left out, as from
the fingerprints. The json says `policy_goal`. The sim2sim node runs such an export
under `--interface human` only.

A fingerprint is the first 8 hex digits of the SHA-256 of the network's weights saved
with `torch.save` (per-robot entries left out, tensors in name order). For the policy
those are exactly the bytes written by `--export-weights`, so `sha256sum` on the
written files shows the same digits without trusting this script.

  uv run --group export python -m deploy.export_onnx <run_dir> <out_dir> [--iteration N] \
    [--goal-anchor robot]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from morphmimic.constants import (
  ACTION_DIM,
  GOAL_DIM,
  GOAL_K,
  GOAL_VERSION,
  HUMAN_FEAT_DIM,
  HUMAN_HEADING_DIM,
  JOINT_NAMES,
  NUM_HUMAN,
  PROPRIO_DIM,
  GoalAnchor,
)
from morphmimic.model.human import mask_human_goal
from morphmimic.model.policy import Policy, PolicyCfg
from morphmimic.model.state import rot6d_from_quat
from morphmimic.robots import check_imu_body
from morphmimic.train.imagination import act
from morphmimic.train.normalizers import RunningNormalizer, normalize, normalize_history
from morphmimic.train.trainer import (
  _stale_kp_parts,
  _stale_kp_warning,
  _without_stale_kp_parts,
  check_goal_anchor,
)

FINGERPRINT_LEN = 8
OPSET = 18
PARITY_BATCH = 64
PARITY_ATOL = 1e-4
_CKPT = re.compile(r"ckpt_(\d+)\.pt$")
# Policy parameters of the human encoder (keypoint goal version 7), exported when
# present, and of the five-point encoder (versions 1-2), which the export leaves out.
HUMAN_ENCODER = "human_encoder."
FIVE_POINT_ENCODER = "kp_encoder."
# Policy parameters of the direct human policy's reconstruction head (training only),
# which the export leaves out.
RECON_HEAD = "recon_head."
NO_HUMAN = "no human interface (the checkpoint has no human encoder)"

# Entries of a network's state dict that belong to one robot (they differ between the
# checkpoints by design), with the reason. Everything else in a shared network must be
# bit-identical.
PER_ROBOT = {
  "policy": {"decoder.morph_actor": "the robot's morphology vector (json `morph`)"},
  "critic": {"morph": "the robot's morphology vector"},
  "target_critic": {"morph": "the robot's morphology vector"},
  "wm": {
    "morph": "the robot's morphology vector",
    "normalizer.mean": "state normaliser, fit on the robot's motions",
    "normalizer.std": "state normaliser, fit on the robot's motions",
    "height_mean": "body-height statistics of the robot",
    "height_std": "body-height statistics of the robot",
    "residual_scale": "residual scale, estimated from the robot's warm-up",
    "joint_limits": "the robot's joint limits",
    "default_joint_pos": "the robot's default pose",
  },
}
NETWORK_LABEL = {
  "policy": "policy",
  "critic": "critic",
  "target_critic": "target critic",
  "wm": "world model",
}
# Which networks a `--share` mode shares.
SHARED_BY = {
  "all": ("policy", "critic", "target_critic", "wm"),
  "policy": ("policy", "critic", "target_critic"),
  "wm": ("wm",),
}
# Per-robot inputs of the ONNX model: json key -> (checkpoint entry, statistic, width).
ROBOT_INPUTS = {
  "obs_mean": ("obs_norm", "mean", PROPRIO_DIM),
  "obs_var": ("obs_norm", "var", PROPRIO_DIM),
  "goal_mean": ("goal_norm", "mean", GOAL_DIM),
  "goal_var": ("goal_norm", "var", GOAL_DIM),
}
# The same for a checkpoint with a human encoder.
HUMAN_ROBOT_INPUTS = {
  "human_mean": ("human_norm", "mean", HUMAN_FEAT_DIM),
  "human_var": ("human_norm", "var", HUMAN_FEAT_DIM),
  "heading_mean": ("heading_norm", "mean", HUMAN_HEADING_DIM),
  "heading_var": ("heading_norm", "var", HUMAN_HEADING_DIM),
}


@dataclass
class NetworkCheck:
  """One network across the checkpoints."""

  network: str
  shared: bool
  """Shared by the run's `--share` mode, so it must be identical."""
  values: int
  """Number of compared values (per-robot entries left out)."""
  fingerprints: dict[str, str]
  differing: list[str] = field(default_factory=list)

  @property
  def identical(self) -> bool:
    return not self.differing


@dataclass
class Report:
  run: str
  share: str
  iteration: int
  robots: list[str]
  checks: list[NetworkCheck]
  files: dict[str, str]
  """File name -> SHA-256."""
  parity: dict[str, dict[str, float]]
  """Robot -> interface ("motion", "human" when exported) -> max |ONNX - PyTorch| of
  its actions."""

  @property
  def policy_fingerprint(self) -> str:
    return next(c for c in self.checks if c.network == "policy").fingerprints[self.robots[0]]


def resolve_paths(paths: list[Path], iteration: int | None) -> list[Path]:
  """Checkpoint files from `paths`: files are kept; a run directory gives each robot
  subdirectory's `ckpt_<iteration>.pt`, by default the latest iteration that every
  robot has."""
  files: list[Path] = []
  for path in map(Path, paths):
    if path.is_file():
      files.append(path)
      continue
    robots = sorted(d for d in path.iterdir() if d.is_dir() and any(d.glob("ckpt_*.pt")))
    if not robots:
      raise ValueError(f"{path}: no <robot>/ckpt_<it>.pt checkpoints")
    its = [{int(m.group(1)) for f in d.iterdir() if (m := _CKPT.search(f.name))}
           for d in robots]
    common = set.intersection(*its)
    it = iteration if iteration is not None else max(common, default=None)
    if it is None or it not in common:
      raise ValueError(f"{path}: no iteration {iteration} in every robot dir {robots}")
    files += [d / f"ckpt_{it}.pt" for d in robots]
  return files


def _weights(state: dict[str, Tensor], network: str) -> dict[str, Tensor]:
  """The network's tensors without its per-robot entries, in name order, contiguous."""
  own = PER_ROBOT.get(network, {})
  return {k: v.detach().contiguous() for k, v in sorted(state.items()) if k not in own}


def _saved(tensors: dict[str, Tensor]) -> bytes:
  buf = io.BytesIO()
  torch.save(tensors, buf)
  return buf.getvalue()


def _fingerprint(tensors: dict[str, Tensor]) -> str:
  return hashlib.sha256(_saved(tensors)).hexdigest()[:FINGERPRINT_LEN]


def _exported_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
  """A policy state dict without the five-point encoder (`FIVE_POINT_ENCODER`) and the
  reconstruction head (`RECON_HEAD`)."""
  return {k: v for k, v in state.items()
          if not k.startswith((FIVE_POINT_ENCODER, RECON_HEAD))}


def has_human(ckpt: dict) -> bool:
  """Whether the checkpoint's policy has a human encoder (`human_encoder.*`)."""
  return any(k.startswith(HUMAN_ENCODER) for k in ckpt["policy"])


def human_policy(ckpt: dict) -> bool:
  """Whether the checkpoint is a direct human policy (`policy_goal="human"`): its
  policy reads the human goal only."""
  return ckpt["cfg"].get("policy_goal", "motion") == "human"


def _motion_only(ckpt: dict, path: Path) -> dict:
  """The checkpoint as the export reads it: its motion interface and its human one, if
  any. Keypoint parts of an older keypoint goal version are dropped with the warning
  `Trainer.load` gives: the five-point interface (versions 1-2: `kp_encoder.*`,
  `kp_norm`, the five-point critics) and the human interface of versions 3-6 (relative
  to the robot pelvis, without the heading error, with the yaw-only one, or with the
  block fed raw: `human_encoder.*`, `human_norm`, `heading_norm`, `critic_kp`), so
  such a checkpoint exports no `action_human`. A human encoder needs its normalisers
  `human_norm` and `heading_norm` (ValueError)."""
  stale = _stale_kp_parts(ckpt)
  if stale is not None:
    warnings.warn(_stale_kp_warning(path, ckpt, stale), stacklevel=2)
    ckpt = _without_stale_kp_parts(ckpt, stale)
  if has_human(ckpt):
    missing = [k for k in ("human_norm", "heading_norm") if k not in ckpt]
    if missing:
      raise ValueError(
        f"checkpoint {path}: a human encoder without its normaliser {', '.join(missing)}"
      )
  return {**ckpt, "policy": _exported_state(ckpt["policy"])}


def _load(paths: list[Path]) -> dict[str, dict]:
  ckpts: dict[str, dict] = {}
  for path in paths:
    ckpt = _motion_only(torch.load(path, map_location="cpu", weights_only=False), path)
    robot = ckpt["cfg"]["robot"]
    if robot in ckpts:
      raise ValueError(f"two checkpoints of robot {robot}: {path}")
    ckpts[robot] = ckpt
    ckpt["_path"] = str(path)
  first = next(iter(ckpts.values()))
  for key in ("iteration", "goal_version"):
    values = {r: c.get(key) for r, c in ckpts.items()}
    if len(set(values.values())) != 1:
      raise ValueError(f"checkpoints differ in {key}: {values}")
  for robot, ckpt in ckpts.items():
    check_imu_body(ckpt, ckpt["_path"])
  if int(first.get("goal_version", 1)) != GOAL_VERSION:
    raise ValueError(
      f"goal version {first.get('goal_version', 1)}; this code builds {GOAL_VERSION}"
    )
  share = {r: c["cfg"].get("share") for r, c in ckpts.items()}
  if len(set(share.values())) != 1:
    raise ValueError(f"checkpoints differ in share: {share}")
  human = {r: has_human(c) for r, c in ckpts.items()}
  if len(set(human.values())) != 1:
    raise ValueError(f"checkpoints differ in having a human encoder: {human}")
  goals = {r: c["cfg"].get("policy_goal", "motion") for r, c in ckpts.items()}
  if len(set(goals.values())) != 1:
    raise ValueError(f"checkpoints differ in policy_goal: {goals}")
  return ckpts


def check_shared(ckpts: dict[str, dict]) -> list[NetworkCheck]:
  """Compares every network present in all checkpoints."""
  share = next(iter(ckpts.values()))["cfg"]["share"]
  checks = []
  for network in NETWORK_LABEL:
    if not all(network in c for c in ckpts.values()):
      continue
    weights = {r: _weights(c[network], network) for r, c in ckpts.items()}
    first = next(iter(weights.values()))
    differing = sorted(
      name for name in first
      if not all(name in w and torch.equal(w[name], first[name]) for w in weights.values())
    )
    checks.append(NetworkCheck(
      network=network,
      shared=network in SHARED_BY.get(share, ()),
      values=sum(t.numel() for t in first.values()),
      fingerprints={r: _fingerprint(w) for r, w in weights.items()},
      differing=differing,
    ))
  return checks


class DeployPolicy(nn.Module):
  """The shared policy with every per-robot part as an input; the motion interface.

  Same computation as `imagination.act`: inputs normalised with the given statistics,
  the decoder fed the given morphology vector."""

  def __init__(self, policy: Policy, eps: float) -> None:
    super().__init__()
    self.policy = policy
    self.eps = eps

  def forward(
    self,
    obs_history: Tensor,
    goal_motion: Tensor,
    obs_mean: Tensor,
    obs_var: Tensor,
    goal_mean: Tensor,
    goal_var: Tensor,
    morph: Tensor | None = None,
  ) -> Tensor:
    b = obs_history.shape[0]
    obs = normalize(obs_history.reshape(b, -1, PROPRIO_DIM), obs_mean, obs_var, self.eps)
    obs = obs.reshape(b, -1)
    goal = normalize(goal_motion, goal_mean, goal_var, self.eps)
    return self.policy(obs, goal, morph)


class DeployHumanPolicy(DeployPolicy):
  """`DeployPolicy` with the human interface as well, for a policy with a human
  encoder: (action_motion, action_human).

  The human branch is `imagination.act_human` after the feature build: `goal_human`
  [B, 2160] are the human features before normalisation (`human_features_from_goal`,
  built from the raw human poses outside the graph) and `human_heading` [B, 60] the raw
  heading block (`heading_features`); each is normalised with its given statistics,
  then masked (`mask_human_goal`: the bool slot [B, 24] and frame [B, K] masks, and
  `human_heading_mask` [B], which zeroes a row's block, exactly) and fed to
  `forward_human`."""

  def forward(
    self,
    obs_history: Tensor,
    goal_motion: Tensor,
    goal_human: Tensor,
    human_point_mask: Tensor,
    human_frame_mask: Tensor,
    human_heading: Tensor,
    human_heading_mask: Tensor,
    obs_mean: Tensor,
    obs_var: Tensor,
    goal_mean: Tensor,
    goal_var: Tensor,
    human_mean: Tensor,
    human_var: Tensor,
    heading_mean: Tensor,
    heading_var: Tensor,
    morph: Tensor | None = None,
  ) -> tuple[Tensor, Tensor]:
    b = obs_history.shape[0]
    obs = normalize(obs_history.reshape(b, -1, PROPRIO_DIM), obs_mean, obs_var, self.eps)
    obs = obs.reshape(b, -1)
    goal = normalize(goal_motion, goal_mean, goal_var, self.eps)
    feats = normalize(goal_human, human_mean, human_var, self.eps)
    heading = normalize(human_heading, heading_mean, heading_var, self.eps)
    human = mask_human_goal(
      feats, human_point_mask, human_frame_mask, heading, human_heading_mask
    )
    return self.policy(obs, goal, morph), self.policy.forward_human(obs, human, morph)


class DeployHumanOnlyPolicy(DeployPolicy):
  """The direct human policy (`policy_goal="human"`): `DeployHumanPolicy`'s human
  branch alone, action_human; no motion goal or its normaliser."""

  def forward(
    self,
    obs_history: Tensor,
    goal_human: Tensor,
    human_point_mask: Tensor,
    human_frame_mask: Tensor,
    human_heading: Tensor,
    human_heading_mask: Tensor,
    obs_mean: Tensor,
    obs_var: Tensor,
    human_mean: Tensor,
    human_var: Tensor,
    heading_mean: Tensor,
    heading_var: Tensor,
    morph: Tensor | None = None,
  ) -> Tensor:
    b = obs_history.shape[0]
    obs = normalize(obs_history.reshape(b, -1, PROPRIO_DIM), obs_mean, obs_var, self.eps)
    obs = obs.reshape(b, -1)
    feats = normalize(goal_human, human_mean, human_var, self.eps)
    heading = normalize(human_heading, heading_mean, heading_var, self.eps)
    human = mask_human_goal(
      feats, human_point_mask, human_frame_mask, heading, human_heading_mask
    )
    return self.policy.forward_human(obs, human, morph)


def _build_policy(ckpt: dict) -> Policy:
  """The checkpoint's policy, as `load_policy` builds it (no morphology input for a run
  without one, e.g. a single-robot run), with its human encoder when it has one; the
  five-point encoder's and the reconstruction head's parameters (`FIVE_POINT_ENCODER`,
  `RECON_HEAD`) are left out."""
  cfg, morph = ckpt["cfg"], ckpt.get("morph")
  policy = Policy(PolicyCfg(
    enc_hidden=tuple(cfg["policy_hidden"]),
    dec_hidden=tuple(cfg["policy_hidden"]),
    act=cfg["act"],
    hist_len=int(cfg["hist_len"]),
    latent=int(cfg.get("latent", 32)),
    morph_dim=0 if morph is None else int(morph["actor"].numel()),
    film=False if morph is None else bool(morph["film"]),
    human=has_human(ckpt),
  ))
  policy.load_state_dict(_exported_state(ckpt["policy"]))
  return policy.eval()


def _robot_inputs(ckpt: dict) -> dict[str, np.ndarray]:
  """The robot's inputs in `DeployPolicy` / `DeployHumanPolicy` /
  `DeployHumanOnlyPolicy` order (the last without the motion goal's statistics)."""
  motion = {k: v for k, v in ROBOT_INPUTS.items()
            if not (human_policy(ckpt) and v[0] == "goal_norm")}
  spec = {**motion, **(HUMAN_ROBOT_INPUTS if has_human(ckpt) else {})}
  out = {key: ckpt[entry][stat].numpy().astype(np.float32)
         for key, (entry, stat, _) in spec.items()}
  morph = ckpt["policy"].get("decoder.morph_actor")
  if morph is not None and morph.numel():  # a run without the morphology input has none
    out["morph"] = morph.numpy().astype(np.float32)
  return out


def _parity_batch(hist_len: int, human: bool, motion: bool = True) -> dict[str, np.ndarray]:
  """Random per-sample inputs (no motion goal without `motion`: the direct human
  policy); with `human` also random human features and masks (one
  row with every slot and frame visible, one with none), heading blocks (row 0 the
  identity in every frame, row 1 unknown, the rest random rotations, rot6d) and
  heading masks (row 0 on, row 1 off, the rest random)."""
  rng = np.random.default_rng(0)
  b = PARITY_BATCH
  out = {
    "obs_history": rng.normal(size=(b, hist_len * PROPRIO_DIM)).astype(np.float32),
    "goal_motion": rng.normal(size=(b, GOAL_DIM)).astype(np.float32),
  }
  if not motion:
    del out["goal_motion"]
  if human:
    points = rng.random((b, NUM_HUMAN)) < 0.5
    frames = rng.random((b, GOAL_K)) < 0.5
    points[0], frames[0] = True, True
    points[1], frames[1] = False, False
    quat = torch.from_numpy(rng.normal(size=(b, GOAL_K, 4)))
    quat[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    quat = nn.functional.normalize(quat, dim=-1)
    heading = rot6d_from_quat(quat).flatten(1).numpy().astype(np.float32)
    heading[1] = 0
    heading_mask = rng.random(b) < 0.7
    heading_mask[0], heading_mask[1] = True, False
    out |= {
      "goal_human": rng.normal(size=(b, HUMAN_FEAT_DIM)).astype(np.float32),
      "human_point_mask": points,
      "human_frame_mask": frames,
      "human_heading": heading,
      "human_heading_mask": heading_mask,
    }
  return out


def _pytorch_actions(ckpt: dict, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """One robot's actions through the training input path (its own checkpoint):
  "motion" (`act`; not for a direct human policy) and, for a human checkpoint, "human"
  (`act_human` after the feature build)."""
  policy = _build_policy(ckpt)
  dims = {"obs_norm": PROPRIO_DIM, "goal_norm": GOAL_DIM, "human_norm": HUMAN_FEAT_DIM,
          "heading_norm": HUMAN_HEADING_DIM}
  norms = {}
  for entry, dim in dims.items():
    if entry in ckpt:
      norms[entry] = RunningNormalizer(dim).eval()
      norms[entry].load_state_dict(ckpt[entry])
  t = {k: torch.from_numpy(v) for k, v in batch.items()}
  out = {}
  with torch.no_grad():
    if not human_policy(ckpt):
      out["motion"] = act(policy, norms["obs_norm"], norms["goal_norm"], t["obs_history"],
                          t["goal_motion"]).numpy()
    if policy.human_encoder is not None:
      goal = mask_human_goal(norms["human_norm"](t["goal_human"]), t["human_point_mask"],
                             t["human_frame_mask"], norms["heading_norm"](t["human_heading"]),
                             t["human_heading_mask"])
      obs = normalize_history(norms["obs_norm"], t["obs_history"])
      out["human"] = policy.forward_human(obs, goal).numpy()
  return out


def _export_onnx(policy: Policy, eps: float, inputs: dict[str, np.ndarray],
                 batch: dict[str, np.ndarray], path: Path, motion: bool = True) -> None:
  """`motion` False: the direct human policy (`DeployHumanOnlyPolicy`, action_human
  only)."""
  human = policy.human_encoder is not None
  if not motion:
    model = DeployHumanOnlyPolicy(policy, eps).eval()
    outputs = ["action_human"]
  else:
    model = (DeployHumanPolicy if human else DeployPolicy)(policy, eps).eval()
    outputs = ["action_motion", "action_human"] if human else ["action_motion"]
  names = list(batch) + list(inputs)
  args = tuple(torch.from_numpy(batch[k][:2]) for k in batch) + tuple(
    torch.from_numpy(inputs[k]) for k in inputs
  )
  batch_dim = torch.export.Dim("batch")
  dynamic = {k: ({0: batch_dim} if k in batch else None) for k in names}
  program = torch.onnx.export(
    model, args, input_names=names, output_names=outputs,
    dynamic_shapes=dynamic, opset_version=OPSET, dynamo=True, external_data=False,
  )
  # The exporter stores each node's Python stack trace, with this machine's absolute paths.
  for node in program.model.graph.all_nodes():
    node.metadata_props.pop("pkg.torch.onnx.stack_trace", None)
  program.save(str(path), external_data=False)


def _git_commit() -> str:
  try:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=Path(__file__).parent, check=True)
    return out.stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _manifest(report: Report, ckpts: dict[str, dict], hist_len: int) -> str:
  robots = report.robots
  human = has_human(ckpts[robots[0]])
  motion = not human_policy(ckpts[robots[0]])
  lines = [
    "SuperMimic deployment export",
    (f"  run          {report.run} (--share {report.share}), iteration {report.iteration},"
     f" goal version {GOAL_VERSION}"),
    f"  written      {datetime.now():%Y-%m-%d %H:%M} by deploy/export_onnx.py at {_git_commit()}",
    "",
  ]
  if not motion:
    lines += [
      "DIRECT HUMAN POLICY (--policy-goal human): the policy reads the human goal only (its",
      "motion encoder was never trained). action_motion is not exported, and goal_motion,",
      "goal_mean, goal_var are not inputs. The sim2sim node runs it with --interface human",
      "only (policy_node.py, deploy.run).",
      "",
    ]
  if len(robots) == 1:
    lines += [
      f"SINGLE-ROBOT POLICY: the {robots[0]} policy of this checkpoint; policy.onnx holds it.",
      "",
      "Network fingerprints (first 8 hex digits of the SHA-256 of the network's weights,",
      "per-robot entries left out; for the policy, the SHA-256 of the file written by",
      "--export-weights)",
      "",
    ]
  else:
    lines += [
      (f"ONE POLICY: its weights are bit-identical in the {len(robots)} checkpoints"
       f" ({', '.join(robots)}); policy.onnx holds them once."),
      "",
      "Shared-network check (fingerprint = first 8 hex digits of the SHA-256 of the",
      "network's weights, per-robot entries left out; for the policy, the SHA-256 of the",
      "files written by --export-weights, so `sha256sum <robot>.policy.pt` shows it too)",
      "",
    ]
  single = len(robots) == 1
  header = f"  {'network':28s}" + ("" if single else f"{'shared':8s}") + f"{'values':>12s}  "
  lines.append(header + "".join(f"{r:10s}" for r in robots) + ("" if single else "result"))
  for c in report.checks:
    if single:
      result = ""
    elif c.identical:
      result = "identical"
    elif c.shared:
      result = f"DIFFERS: {', '.join(c.differing[:3])}"
    else:
      result = "separate (a network per robot)"
    shared = "" if single else f"{'yes' if c.shared else 'no':8s}"
    lines.append(
      f"  {NETWORK_LABEL[c.network]:28s}{shared}{c.values:>12,d}  "
      + "".join(f"{c.fingerprints[r]:10s}" for r in robots) + result
    )
  lines += ["", "Per-robot entries (differ by design; in <robot>.json):", ""]
  has_morph = "morph" in ckpts[robots[0]]
  if has_morph:
    width = ckpts[robots[0]]["morph"]["actor"].numel()
    lines.append(f"  {'decoder morphology vector':28s}{width:>5d}  json `morph`:"
                 " which body the decoder drives")
  cfg = ckpts[robots[0]]["cfg"]
  pooled = cfg.get("robots") is not None and cfg.get("pool_norms", False)
  fit = "pooled over every robot's data" if pooled else "fit on the robot's own data"
  entries = [("obs_norm", "proprioception", fit), ("goal_norm", "motion goal", fit)]
  if not motion:
    entries = entries[:1]
  if human:  # the trainer always pools the human normalisers over the run's robots
    multi = cfg.get("robots") is not None
    how = "pooled over every robot's data" if multi else "fit on the robot's own data"
    entries.append(("human_norm", "human features", how))
    entries.append(("heading_norm", "heading block", how))
  spec = {**ROBOT_INPUTS, **HUMAN_ROBOT_INPUTS}
  for entry, what, how in entries:
    keys = [k for k, (e, _, _) in spec.items() if e == entry]
    width = spec[keys[0]][2]
    lines.append(f"  {entry + ' mean, var':28s}{width:>5d}  json `{keys[0]}`, `{keys[1]}`:"
                 f" {what} normaliser, {how}")
  if pooled:
    lines.append("  (normalisers pooled across robots, --pool-norms: the same values in"
                 " every json)")
  lines += [
    "  (in the training checkpoints also: the world model's state normaliser, height",
    "  statistics, residual scale, joint limits and default pose; the critics' copies",
    "  of the morphology vector; the reset sampler and optimiser state)",
    "",
    "ONNX model policy.onnx (B = batch)",
    "",
    (f"  obs_history     [B, {hist_len * PROPRIO_DIM}]  float  {hist_len} proprioception"
     f" frames, oldest first, {PROPRIO_DIM} each"),
  ]
  if motion:
    lines.append(f"  goal_motion     [B, {GOAL_DIM}]  float  motion goal")
  if human:
    lines += [
      (f"  goal_human      [B, {HUMAN_FEAT_DIM}]  float  human features before"
       " normalisation (human_features_from_goal,"),
      "                  built outside the graph from the raw human poses)",
      f"  human_point_mask [B, {NUM_HUMAN}]  bool  which human slots are seen",
      f"  human_frame_mask [B, {GOAL_K}]  bool  which goal frames are seen",
      (f"  human_heading   [B, {HUMAN_HEADING_DIM}]  float  the human Spine3's orientation"
       " relative to the robot"),
      (f"                  torso, rot6d per goal frame ({GOAL_K} x 6, heading_features):"
       " R_k = inv(robot torso)"),
      "                  * spine3_k * inv(rest), rest = Rz(pi/2) * Rx(pi/2) (the SMPL",
      "                  rest frame; robot torso = IMU * waist FK); the identity when",
      "                  the torso is upright and faces the way the chest faces; raw",
      "                  (the graph normalises it; without the Spine3 slot the block",
      "                  is the Pelvis's, else the Head's, against the torso); when",
      "                  unknown set human_heading_mask False (the block is then ignored)",
      ("  human_heading_mask [B]  bool  whether the heading block is known; False"
       " zeroes the"),
      "                  row's block after the normalisation (exact zeros)",
    ]
  lines += [
    "  obs_mean, obs_var"
    + (", goal_mean, goal_var" if motion else "")
    + (", human_mean, human_var, heading_mean, heading_var" if human else "")
    + (", morph" if has_morph else " (no morphology input in this run)"),
    "                  [width]  float  the robot's values, from <robot>.json",
  ]
  if motion:
    lines.append(f"  -> action_motion [B, {ACTION_DIM}]  through the motion goal")
  if not motion:
    lines += [
      (f"  -> action_human [B, {ACTION_DIM}]  through the human skeleton goal (human_norm,"
       " heading_norm, mask_human_goal,"),
      "     the human encoder); the only output (--policy-goal human)",
    ]
  elif human:
    lines += [
      (f"  -> action_human [B, {ACTION_DIM}]  through the human skeleton goal (human_norm,"
       " heading_norm, mask_human_goal,"),
      "     the human encoder); the sim2sim node reads it with --interface human",
    ]
  else:
    lines.append(f"  {NO_HUMAN}")
  lines += [
    (f"  Actions and joint observations use the canonical {len(JOINT_NAMES)}-joint order"
     " (the G1's;"),
    "  `canonical_joint_order` in the json); other robots map it with their translator.",
    "",
    f"Parity with PyTorch (each robot's own checkpoint, {PARITY_BATCH} random inputs):",
    "",
  ]
  for r, diffs in report.parity.items():
    lines.append(f"  {r:6s} max |ONNX - PyTorch|  "
                 + "  ".join(f"{name} {d:.1e}" for name, d in diffs.items()))
  lines += ["", "Files (check with: sha256sum -c SHA256SUMS)", ""]
  lines += [f"  {digest}  {name}" for name, digest in report.files.items()]
  lines += ["", "Source checkpoints", ""]
  lines += [f"  {r:6s} {ckpts[r]['_path']}" for r in robots]
  return "\n".join(lines) + "\n"


def export(paths: list[Path], out_dir: Path, weights_dir: Path | None = None,
           goal_anchor: GoalAnchor = "reference") -> Report:
  """Checks the shared networks of `paths` (one checkpoint per robot), then writes
  `policy.onnx`, `<robot>.json` and `MANIFEST.txt` to `out_dir`. One checkpoint exports
  that robot's policy. Raises ValueError, before writing anything, when several
  checkpoints do not share their policy or a shared network differs, or when a
  checkpoint was trained with another goal anchor than `goal_anchor`
  (`check_goal_anchor`), which the json records. With `weights_dir`, also writes each
  robot's policy weights (per-robot entries left out) to `<robot>.policy.pt`:
  byte-identical files."""
  ckpts = _load(paths)
  for ckpt in ckpts.values():
    check_goal_anchor(ckpt["cfg"], goal_anchor, ckpt["_path"])
  robots = list(ckpts)
  first = ckpts[robots[0]]
  share = first["cfg"]["share"]
  if len(robots) > 1 and "policy" not in SHARED_BY.get(share, ()):
    raise ValueError(
      f"several checkpoints need a run with a shared policy (--share all or policy);"
      f" got {len(robots)} checkpoints, share {share!r}. Export one robot's checkpoint"
      " to get its own policy."
    )
  checks = check_shared(ckpts)
  bad = [c for c in checks if c.shared and not c.identical]
  if bad:
    detail = "; ".join(f"{c.network}: {', '.join(c.differing[:5])}" for c in bad)
    raise ValueError(f"shared networks differ across the checkpoints: {detail}")

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  hist_len = int(first["cfg"]["hist_len"])
  eps = RunningNormalizer(1).eps
  human = has_human(first)
  motion = not human_policy(first)
  batch = _parity_batch(hist_len, human, motion)
  inputs = {r: _robot_inputs(c) for r, c in ckpts.items()}
  _export_onnx(_build_policy(first), eps, inputs[robots[0]], batch, out_dir / "policy.onnx",
               motion)

  import onnxruntime as ort

  session = ort.InferenceSession(str(out_dir / "policy.onnx"))
  outputs = [o.name for o in session.get_outputs()]
  parity = {}
  for r, ckpt in ckpts.items():
    got = dict(zip(outputs, session.run(None, {**batch, **inputs[r]}), strict=True))
    want = _pytorch_actions(ckpt, batch)
    parity[r] = {name: float(np.abs(got[f"action_{name}"] - a).max())
                 for name, a in want.items()}
  if max(d for diffs in parity.values() for d in diffs.values()) > PARITY_ATOL:
    raise ValueError(f"ONNX does not match PyTorch: {parity}")

  robots_dir = Path(__file__).parent / "sim2sim" / "robots"
  for r, ckpt in ckpts.items():
    constants = robots_dir / f"{r}.json"
    if not constants.exists():
      raise ValueError(f"no robot constants {constants}: run deploy/sim2sim/export_assets.py --bake")
    doc = {
      "about": "Per-robot inputs of policy.onnx (see MANIFEST.txt).",
      "run": first["cfg"].get("run_name"),
      "iteration": int(first["iteration"]),
      "goal_version": GOAL_VERSION,
      "kp_goal_version": int(ckpt.get("kp_goal_version", 1)),
      "goal_anchor": goal_anchor,
      "human_interface": human,
      "policy_goal": first["cfg"].get("policy_goal", "motion"),
      "source_checkpoint": ckpt["_path"],
      "canonical_joint_order": list(JOINT_NAMES),
      "inputs": {k: v.tolist() for k, v in inputs[r].items()},
      "robot": json.loads(constants.read_text()),
    }
    (out_dir / f"{r}.json").write_text(json.dumps(doc, indent=1) + "\n")
  if weights_dir is not None:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    for r, ckpt in ckpts.items():
      (weights_dir / f"{r}.policy.pt").write_bytes(_saved(_weights(ckpt["policy"], "policy")))

  names = ["policy.onnx", *(f"{r}.json" for r in robots)]
  report = Report(
    run=str(first["cfg"].get("run_name")),
    share=share,
    iteration=int(first["iteration"]),
    robots=robots,
    checks=checks,
    files={n: hashlib.sha256((out_dir / n).read_bytes()).hexdigest() for n in names},
    parity=parity,
  )
  (out_dir / "MANIFEST.txt").write_text(_manifest(report, ckpts, hist_len))
  (out_dir / "SHA256SUMS").write_text(
    "".join(f"{digest}  {name}\n" for name, digest in report.files.items())
  )
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("paths", nargs="+", type=Path,
                      help="a run directory (<robot>/ckpt_<it>.pt) or checkpoint files")
  parser.add_argument("out_dir", type=Path)
  parser.add_argument("--iteration", type=int, default=None,
                      help="checkpoint iteration (default: the latest every robot has)")
  parser.add_argument("--export-weights", type=Path, default=None,
                      help="also write <robot>.policy.pt weight files here")
  parser.add_argument("--goal-anchor", choices=["reference", "robot"], default="reference",
                      help="the goal anchor the checkpoints were trained with")
  args = parser.parse_args()
  export(resolve_paths(args.paths, args.iteration), args.out_dir, args.export_weights,
         args.goal_anchor)
  print((args.out_dir / "MANIFEST.txt").read_text(), end="")


if __name__ == "__main__":
  main()
