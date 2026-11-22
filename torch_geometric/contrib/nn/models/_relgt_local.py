# Adapted from snap-stanford/relgt, local_module.py, commit
# 19e423ca3e7cac761130aba790857f2dc3a46ef7 (MIT).
# https://github.com/snap-stanford/relgt
# See relgt.LICENSE in this directory for the original license notice.

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    GELU,
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Sequential,
)


class _RelGTFeedForward(Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.bn_in = BatchNorm1d(channels)
        self.bn_out = BatchNorm1d(channels)
        self.ffn_net = Sequential(
            Linear(channels, 2 * channels),
            GELU(),
            Dropout(dropout),
            Linear(2 * channels, channels),
            Dropout(dropout),
        )

    def reset_parameters(self) -> None:
        self.bn_in.reset_parameters()
        self.bn_out.reset_parameters()
        for layer in self.ffn_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn_in(x.transpose(1, 2)).transpose(1, 2)
        x = self.ffn_net(x)
        return self.bn_out(x.transpose(1, 2)).transpose(1, 2)


class _RelGTEncoderLayer(Module):
    def __init__(self, channels: int, heads: int, dropout: float,
                 attn_dropout: float):
        super().__init__()
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.self_attention_norm = LayerNorm(channels)
        self.q_proj = Linear(channels, channels)
        self.k_proj = Linear(channels, channels)
        self.v_proj = Linear(channels, channels)
        self.out_proj = Linear(channels, channels)
        self.self_attention_dropout = Dropout(dropout)
        self.ffn_norm = LayerNorm(channels)
        self.ffn = _RelGTFeedForward(channels, dropout)

    def reset_parameters(self) -> None:
        self.self_attention_norm.reset_parameters()
        for projection in [
                self.q_proj, self.k_proj, self.v_proj, self.out_proj
        ]:
            torch.nn.init.xavier_uniform_(projection.weight)
            torch.nn.init.zeros_(projection.bias)
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        batch_size, context_size, channels = x.shape
        normalized = self.self_attention_norm(x)
        head_channels = channels // self.heads
        q, k, v = (projection(normalized).reshape(batch_size, context_size,
                                                  self.heads,
                                                  head_channels).transpose(
                                                      1, 2)
                   for projection in [self.q_proj, self.k_proj, self.v_proj])
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch_size, context_size,
                                                    channels)
        x = x + self.self_attention_dropout(self.out_proj(attended))
        return x + self.ffn(self.ffn_norm(x))


class _RelGTLocal(Module):
    """Reference RelGT context attention and seed-plus-neighbor readout.

    The outer branch normalization and prediction head belong to RelGT.
    Input contexts have equal length, with the seed at position zero.
    """
    def __init__(self, channels: int, num_layers: int = 4, heads: int = 4,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        for name, value in [('channels', channels), ('num_layers', num_layers),
                            ('heads', heads)]:
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(f"'{name}' must be a positive integer")
        if channels % heads != 0:
            raise ValueError("'channels' must be divisible by 'heads'")
        for name, value in [('dropout', dropout),
                            ('attn_dropout', attn_dropout)]:
            if not 0.0 <= value < 1.0:
                raise ValueError(f"'{name}' must be in [0, 1)")

        self.channels = channels
        self.num_layers = num_layers
        self.heads = heads
        self.att_embeddings_nope = Linear(channels, channels)
        self.layers = ModuleList([
            _RelGTEncoderLayer(channels, heads, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = LayerNorm(channels)
        self.attn_layer = Linear(2 * channels, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.att_embeddings_nope.reset_parameters()
        self.final_ln.reset_parameters()
        self.attn_layer.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 3 or tokens.size(-1) != self.channels:
            raise ValueError("'tokens' must have shape [B, K, channels]")
        if tokens.size(0) == 0 or tokens.size(1) == 0:
            raise ValueError(
                "'tokens' must contain at least one context/token")
        if not tokens.is_floating_point():
            raise TypeError("'tokens' must be floating point")
        if not torch.isfinite(tokens).all():
            raise ValueError("'tokens' must be finite")
        if self.training and tokens.size(0) * tokens.size(1) < 2:
            raise ValueError(
                "Training requires at least two token occurrences")

        x = self.att_embeddings_nope(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        seed = x[:, 0]
        if x.size(1) == 1:
            return seed
        neighbors = x[:, 1:]
        repeated_seed = seed.unsqueeze(1).expand_as(neighbors)
        scores = self.attn_layer(torch.cat([repeated_seed, neighbors], dim=-1))
        weights = scores.softmax(dim=1)
        return seed + (weights * neighbors).sum(dim=1)
