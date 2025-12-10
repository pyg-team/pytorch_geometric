from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


class KeypointAttention(nn.Module):
    r"""
    Single-headed keypoint attention over nodes with 3D coordinates.

    Given:
        - node features x ∈ ℝ^{N × d}
        - node positions pos ∈ ℝ^{N × 3}

    this module learns K query vectors and computes attention over the nodes
    to produce K keypoints:

        attn_k   = softmax( q_k · W_k x_i )
        kp_feat_k = Σ_i attn_k[i] * W_v x_i
        kp_pos_k  = Σ_i attn_k[i] * pos_i

    Properties:
        - Feature space: invariant to global SE(3) transforms, since attention
          depends only on x.
        - Coordinate space: equivariant to SE(3). If `pos` is rotated and/or
          translated, `kp_pos` transforms accordingly because it is a convex
          combination of the input coordinates.

    This is suitable for EquiBind-style keypoints: learnable global summaries
    used for subsequent Kabsch alignment.
    """

    def __init__(
        self,
        num_keypoints: int,
        node_dim: int,
        attn_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_keypoints = num_keypoints
        self.node_dim = node_dim
        self.attn_dim = attn_dim if attn_dim is not None else node_dim
        self.out_dim = out_dim if out_dim is not None else node_dim

        # Learnable keypoint queries: Q ∈ ℝ^{K × attn_dim}
        self.query = nn.Parameter(torch.randn(num_keypoints, self.attn_dim))

        # Linear projections for keys and values from node features
        self.key_proj = nn.Linear(node_dim, self.attn_dim)
        self.value_proj = nn.Linear(node_dim, self.out_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, x: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Args:
            x:   Node features of shape [N, d].
            pos: Node coordinates of shape [N, 3].

        Returns:
            kp_pos:  Keypoint coordinates [K, 3].
            kp_feat: Keypoint features [K, out_dim].
            attn:    Attention weights [K, N] over nodes for each keypoint.
        """
        if x.dim() != 2 or pos.dim() != 2:
            raise ValueError(
                f"x and pos must be [N, d] and [N, 3], got {x.shape} and {pos.shape}"
            )
        if x.size(0) != pos.size(0):
            raise ValueError(
                f"x and pos must have the same number of nodes, got {x.size(0)} and {pos.size(0)}"
            )

        N = x.size(0)
        K = self.num_keypoints

        # Keys and values from node features
        keys = self.key_proj(x)            # [N, attn_dim]
        values = self.value_proj(x)        # [N, out_dim]

        # Learnable queries
        Q = self.query                     # [K, attn_dim]

        # Attention scores and weights: [K, N]
        scores = Q @ keys.transpose(-1, -2) / math.sqrt(self.attn_dim)
        attn = torch.softmax(scores, dim=-1)

        # Keypoint features and positions
        kp_feat = attn @ values            # [K, out_dim]
        kp_pos = attn @ pos                # [K, 3]

        return kp_pos, kp_feat, attn
