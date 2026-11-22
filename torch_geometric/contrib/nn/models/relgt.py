# Adapted from https://github.com/snap-stanford/relgt (MIT).
# See relgt.LICENSE for the original copyright and license notice.
# The released EMA code also credits https://github.com/devnkong/GOAT.
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, LayerNorm, Linear

from torch_geometric.contrib.nn.models._relgt_local import _RelGTLocal
from torch_geometric.nn import MLP
from torch_geometric.nn.inits import reset


class _RelGTMemory(torch.nn.Module):
    """EMA clustering with persistent per-entity population assignments."""
    def __init__(self, num_nodes: int, num_centroids: int, channels: int,
                 decay: float):
        super().__init__()
        self.decay = decay
        self.num_centroids = num_centroids
        self.norm = BatchNorm1d(channels, affine=False)
        self.register_buffer('centroids', torch.empty(num_centroids, channels))
        self.register_buffer('output', torch.empty(num_centroids, channels))
        self.register_buffer('ema_count', torch.empty(num_centroids))
        self.register_buffer('ema_sum', torch.empty(num_centroids, channels))
        self.register_buffer('assignment',
                             torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer('population',
                             torch.empty(num_centroids, dtype=torch.long))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.centroids.normal_()
        self.output.normal_()
        self.ema_count.zero_()
        self.ema_sum.normal_()
        self.assignment.random_(self.num_centroids)
        self.population.copy_(
            self.assignment.bincount(minlength=self.num_centroids))

    @torch.no_grad()
    def update(self, x: Tensor, seed_id: Tensor) -> None:
        if not self.training:
            return
        x = self.norm(x.detach())
        distance = (x.square().sum(-1, keepdim=True) +
                    self.centroids.square().sum(-1) -
                    2 * x @ self.centroids.t())
        index = distance.argmin(-1)
        counts = index.bincount(minlength=self.num_centroids).to(x.dtype)
        sums = x.new_zeros(self.centroids.shape).index_add_(0, index, x)
        self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        total = self.ema_count.sum()
        # The released implementation stores the smoothed counts each step.
        self.ema_count.copy_((self.ema_count + 1e-5) /
                             (total + self.num_centroids * 1e-5) * total)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        self.centroids.copy_(self.ema_sum / self.ema_count.unsqueeze(-1))
        self.output.copy_(self.centroids *
                          (self.norm.running_var + 1e-5).sqrt() +
                          self.norm.running_mean)

        # Last occurrence wins for assignment; all occurrences enter the EMA.
        # Avoid repeated-index writes, whose order is unspecified on CUDA.
        entities, inverse = seed_id.unique(sorted=True, return_inverse=True)
        last = torch.full_like(entities, -1)
        last.scatter_reduce_(
            0, inverse, torch.arange(seed_id.numel(), device=seed_id.device),
            reduce='amax')
        old, new = self.assignment[entities], index[last]
        self.population.sub_(old.bincount(minlength=self.num_centroids))
        self.population.add_(new.bincount(minlength=self.num_centroids))
        self.assignment[entities] = new


class _RelGTGlobal(torch.nn.Module):
    def __init__(self, channels: int, global_channels: int, num_nodes: int,
                 num_centroids: int, heads: int, decay: float,
                 attn_dropout: float):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.project = Linear(channels, global_channels)
        self.query = Linear(global_channels, channels)
        self.key = Linear(global_channels, channels)
        self.value = Linear(global_channels, channels)
        self.memory = _RelGTMemory(num_nodes, num_centroids, global_channels,
                                   decay)

    def reset_parameters(self) -> None:
        for module in [
                self.project, self.query, self.key, self.value, self.memory
        ]:
            module.reset_parameters()

    def forward(self, seed: Tensor, seed_id: Optional[Tensor]) -> Tensor:
        x = self.project(seed)
        # Copies protect backward from the subsequent in-place EMA update.
        centers = self.memory.output.detach().clone()
        population = self.memory.population.detach().clone()
        if bool((population < 0).any()) or int(population.sum()) == 0:
            raise ValueError('Global memory needs a nonnegative, nonzero '
                             'population')
        batch, count = x.size(0), centers.size(0)
        width = self.channels // self.heads
        q = self.query(x).view(batch, self.heads, width).transpose(0, 1)
        k = self.key(centers).view(count, self.heads, width).transpose(0, 1)
        v = self.value(centers).view(count, self.heads, width).transpose(0, 1)
        # Reference scale is total width, unlike local per-head attention.
        weights = q @ k.transpose(-1, -2) / math.sqrt(self.channels)
        weights = (weights + population.float().log()).softmax(-1)
        weights = F.dropout(weights, self.attn_dropout, self.training)
        out = (weights @ v).transpose(0, 1).reshape(batch, self.channels)
        if self.training:
            self.memory.update(x, seed_id)
        return out


class RelGT(torch.nn.Module):
    r"""The local/global relational transformer from the
    `Relational Graph Transformer <https://arxiv.org/abs/2505.10960>`_ paper.

    Consumes tokens produced by :class:`RelGTTokenEncoder`, with the prediction
    seed at position zero. Returns raw scores of shape
    :obj:`[batch_size, out_channels]`. Dataset loading and losses are external.

    The centroid memory follows the released implementation: assignments cover
    all :obj:`num_nodes` entities and initialize randomly. Training updates the
    memory after reading it; evaluation changes no state. Temporal filtering of
    local contexts alone does not make global memory historically valid: the
    caller must control training order and memory provenance for such claims.

    Initial validation targets float32. Mixed precision is not supported yet;
    centroid statistics can have a large dynamic range during initialization.

    Args:
        channels (int): Token width, divisible by :obj:`heads`.
        out_channels (int): Output width; one for binary or regression tasks.
        num_nodes (int): Number of globally indexed database entities.
        num_layers (int, optional): Local depth. (default: :obj:`4`)
        heads (int, optional): Attention heads. (default: :obj:`4`)
        num_centroids (int, optional): Memory size. (default: :obj:`4096`)
        global_channels (int, optional): Centroid width. Defaults to half the
            token width. (default: :obj:`None`)
        mode (str, optional): :obj:`"full"` or :obj:`"local"`.
            (default: :obj:`"full"`)
        dropout (float, optional): Feed-forward dropout. (default: :obj:`0.`)
        attn_dropout (float, optional): Attention dropout. (default: :obj:`0.`)
        ema_decay (float, optional): Centroid EMA decay. (default: :obj:`0.99`)
    """
    def __init__(self, channels: int, out_channels: int, num_nodes: int,
                 num_layers: int = 4, heads: int = 4,
                 num_centroids: int = 4096,
                 global_channels: Optional[int] = None, mode: str = 'full', *,
                 dropout: float = 0., attn_dropout: float = 0.,
                 ema_decay: float = 0.99):
        super().__init__()
        global_channels = (channels //
                           2 if global_channels is None else global_channels)
        if min(channels, out_channels, num_layers, heads, num_centroids,
               global_channels) <= 0:
            raise ValueError('Dimensions, heads and depth must be positive')
        if num_nodes < 0 or (mode == 'full' and num_nodes == 0):
            raise ValueError("Full mode requires positive 'num_nodes'")
        if channels % heads:
            raise ValueError("'channels' must be divisible by 'heads'")
        if mode not in {'full', 'local'}:
            raise ValueError("'mode' must be 'full' or 'local'")
        if not 0 <= dropout < 1 or not 0 <= attn_dropout < 1:
            raise ValueError('Dropout probabilities must be in [0, 1)')
        if not 0 <= ema_decay < 1:
            raise ValueError("'ema_decay' must be in [0, 1)")
        self.channels = channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.mode = mode
        self.local = _RelGTLocal(channels, num_layers, heads, dropout,
                                 attn_dropout)
        self.local_norm = LayerNorm(channels)
        self.global_module = None
        if mode == 'full':
            self.global_module = _RelGTGlobal(channels, global_channels,
                                              num_nodes, num_centroids, heads,
                                              ema_decay, attn_dropout)
        self.global_norm = LayerNorm(channels) if mode == 'full' else None
        in_channels = channels * (2 if mode == 'full' else 1)
        self.fusion = torch.nn.Sequential(BatchNorm1d(in_channels),
                                          Linear(in_channels, 2 * channels),
                                          torch.nn.GELU(),
                                          torch.nn.Dropout(dropout),
                                          Linear(2 * channels, channels),
                                          torch.nn.Dropout(dropout),
                                          BatchNorm1d(channels))
        self.head = MLP(in_channels=channels, hidden_channels=channels,
                        out_channels=out_channels, num_layers=2)

    def reset_parameters(self) -> None:
        r"""Resets learned parameters, normalization and centroid memory."""
        reset(self.local)
        reset(self.local_norm)
        reset(self.global_module)
        reset(self.global_norm)
        reset(self.fusion)
        reset(self.head)

    def reset_memory(self) -> None:
        r"""Resets global memory and its normalization, preserving weights.

        This alone does not enforce chronological training or establish
        historical feature availability.
        """
        if self.global_module is not None:
            self.global_module.memory.reset_parameters()

    def forward(self, tokens: Tensor,
                seed_id: Optional[Tensor] = None) -> Tensor:
        r"""Predicts from :obj:`[B, K, channels]` tokens and global seed IDs.

        Training requires at least two samples because the reference fusion
        and prediction head use batch normalization. Evaluation supports one.
        :obj:`seed_id` is required during full-mode training. Frozen evaluation
        does not need IDs, permitting predictions for previously unseen seeds.
        """
        if tokens.dim() != 3 or tokens.size(-1) != self.channels:
            raise ValueError("'tokens' must have shape [B, K, channels]")
        if tokens.size(0) == 0 or tokens.size(1) == 0:
            raise ValueError('Batch and context must be nonempty')
        if not tokens.is_floating_point() or not bool(tokens.isfinite().all()):
            raise ValueError("'tokens' must contain finite floating values")
        if self.training and tokens.size(0) < 2:
            raise ValueError('Training batch normalization needs B >= 2')
        if self.global_module is not None:
            if self.training and seed_id is None:
                raise ValueError("'seed_id' is required in full-mode training")
            if seed_id is not None and (seed_id.dtype != torch.long
                                        or seed_id.shape != (tokens.size(0), )
                                        or seed_id.device != tokens.device):
                raise ValueError("'seed_id' must be long [B] on token device")
            invalid_id = seed_id is not None and bool(
                (seed_id < 0).any() or (seed_id >= self.num_nodes).any())
            if invalid_id:
                raise ValueError("'seed_id' is outside [0, num_nodes)")
        local = self.local_norm(self.local(tokens))
        if self.global_module is not None:
            global_out = self.global_norm(
                self.global_module(tokens[:, 0], seed_id))
            local = torch.cat([local, global_out], dim=-1)
        return self.head(self.fusion(local))
