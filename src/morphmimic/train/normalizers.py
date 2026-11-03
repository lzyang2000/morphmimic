"""Running input normalisers of the policy.

The paper does not state how policy inputs are normalised; we use mjlab/BeyondMimic's
rsl_rl default, `EmpiricalNormalization`: per-dim running mean and variance over all
samples seen, updated from the noisy `policy` obs and `goal` during collection and
used as constants everywhere else (imagination differentiates through the affine map).

Data parallel (`sync=True`, one normaliser copy per rank): every update folds the union
of all ranks' batches, so the copies hold the statistics of every sample collected on
any rank and stay identical (the ranks start from the same, broadcast, statistics).
`pool_ranks_` merges copies that each fit their own samples into one pooled set (a
multi-robot resume that turns `TrainCfg.pool_norms` on).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn


class RunningNormalizer(nn.Module):
  """x -> (x - mean) / (std + eps) with per-dim running statistics over `dim`.

  `update(x)` folds a batch [..., dim] into the exact running mean and population
  variance of every sample seen (Chan et al.'s parallel update); it is a no-op in eval
  mode. `mean`, `var` and `count` are buffers, so they move and save with the module.

  With `sync=True` (data parallel; needs a process group) `update` is collective: every
  rank calls it once per step (in training mode), and it folds the pooled batch of all
  ranks (`_pooled_moments`, one all-reduce, no host sync); a rank's batch may be empty.
  """

  mean: Tensor
  var: Tensor
  count: Tensor

  def __init__(self, dim: int, eps: float = 1e-2, sync: bool = False) -> None:
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.sync = sync
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros((), dtype=torch.long))

  @torch.no_grad()
  def update(self, x: Tensor) -> None:
    if not self.training:
      return
    x = x.reshape(-1, self.dim).to(self.mean.dtype)
    if self.sync:
      n, batch_mean, batch_var = self._pooled_moments(x)
      self.count += n
      # A pooled batch of 0 samples leaves the statistics unchanged (rate 0).
      rate = (n / self.count.clamp(min=1).double()).to(self.mean.dtype)
    else:
      n = x.shape[0]
      if n == 0:
        return
      batch_var, batch_mean = torch.var_mean(x, dim=0, unbiased=False)
      self.count += n
      rate = (n / self.count.double()).to(self.mean.dtype)
    delta = batch_mean - self.mean
    self.mean += rate * delta
    # var' = (1 - r) var + r var_b + r (1 - r) delta^2.
    self.var += rate * (batch_var - self.var + delta * (batch_mean - self.mean))

  def _pooled_moments(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(count [long], mean, population variance) of the union of every rank's batch
    `x`: one all-reduce (sum, float64) of each rank's count and first two moments
    about the running mean c (identical on all ranks, close to the data, so the
    variance E[(x - c)^2] - E[x - c]^2 does not cancel)."""
    c = self.mean.double()
    n = x.shape[0]
    if n > 0:
      batch_var, batch_mean = torch.var_mean(x, dim=0, unbiased=False)
      d = batch_mean.double() - c
      stats = torch.cat([c.new_full((1,), n), n * d, n * (batch_var.double() + d * d)])
    else:
      stats = c.new_zeros(1 + 2 * self.dim)
    dist.all_reduce(stats)
    total = stats[0]
    shift, square = stats[1:].view(2, self.dim) / total.clamp(min=1.0)
    var = (square - shift * shift).clamp(min=0.0)
    dtype = self.mean.dtype
    return total.round().long(), (c + shift).to(dtype), var.to(dtype)

  @torch.no_grad()
  def pool_ranks_(self) -> None:
    """Collective (needs a process group): replaces every rank's statistics by those
    of the union of all ranks' samples (counts summed; two all-reduces, float64), so
    copies that each fit their own data become one pooled set. A union without
    samples leaves the statistics as they are. Once, when a resume turns pooling on
    (`TrainCfg.pool_norms`)."""
    n = self.count.double()
    first = torch.cat([n.reshape(1), n * self.mean.double()])
    dist.all_reduce(first)
    total = first[0]
    if total == 0:  # the same on every rank: none takes the second all-reduce
      return
    mean = first[1:] / total
    d = self.mean.double() - mean
    second = n * (self.var.double() + d * d)
    dist.all_reduce(second)
    self.mean.copy_(mean)
    self.var.copy_(second / total)
    self.count.copy_(total.round().long())

  def forward(self, x: Tensor) -> Tensor:
    return normalize(x, self.mean, self.var, self.eps)


def normalize(x: Tensor, mean: Tensor, var: Tensor, eps: float) -> Tensor:
  """(x - mean) / (sqrt(var) + eps): `RunningNormalizer.forward` with explicit
  statistics, for an export whose statistics are inputs (`deploy/export_onnx.py`)."""
  return (x - mean) / (var.sqrt() + eps)


def normalize_history(norm: RunningNormalizer, hist_flat: Tensor) -> Tensor:
  """Flattened obs history [B, H * dim] (`ObsHistory.flat`) normalised frame by frame
  with the single-frame normaliser `norm`."""
  b = hist_flat.shape[0]
  return norm(hist_flat.reshape(b, -1, norm.dim)).reshape(b, -1)
