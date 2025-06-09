"""Learnable spatial bias provider for graph attention based on Graphormer's
design. It adds learnable spatial biases to the attention logits based
on graph distances between nodes.

Ref:

[Ying et al., 2021] Chengxuan Ying, Tianle Cai, Shengjie
 Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen,
 and Tie-Yan Liu. Do transformers really perform bad for
 graph representation? arXiv preprint arXiv:2106.05234,
 2021.

 Code adapted from:
 https://github.com/qwerfdsaplking/Graph-Trans
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

from torch_geometric.contrib.nn.bias.base import BaseBiasProvider
from torch_geometric.data import Data


class GraphAttnSpatialBias(BaseBiasProvider):
    """Learnable spatial bias provider for graph attention.

    Maps integer distances in `data.spatial_pos` to per-head additive
    biases.

    Supports:
      • Single-graph Data with spatial_pos ⟶ (L, L)
      • Batched Data with spatial_pos cat’d as (N, L) plus `ptr`.

    Args:
        num_heads (int): Number of attention heads.
        num_spatial (int): Number of spatial distances (excl. super node).
        use_super_node (bool): If True, reserve one extra index for
            any attention to/from a “super” node at row/col 0.
    """
    def __init__(
        self,
        num_heads: int,
        num_spatial: int,
        use_super_node: bool = False,
    ) -> None:
        super().__init__(num_heads, use_super_node)
        total_tokens = num_spatial + (1 if use_super_node else 0)
        self.super_idx = total_tokens - 1 if use_super_node else None
        self.spatial_embeddings = nn.Embedding(total_tokens, num_heads)
        init.xavier_uniform_(self.spatial_embeddings.weight)

    def _extract_raw_distances(self, data: Data) -> List[Tensor]:
        """Extract per-graph distance blocks from a block-diagonal matrix.

        Given `data.spatial_pos`, a 2D LongTensor of shape (∑Ni, ∑Ni),
        and `data.ptr`, a LongTensor of length B+1 holding cumulative node
        counts, split `spatial_pos` into B separate distance matrices:

        - Let Ni = ptr[i+1] − ptr[i] for i in [0..B−1].
        - Return a list of B tensors, where the i-th tensor is
            spatial_pos[ptr[i]:ptr[i+1], ptr[i]:ptr[i+1]] of shape (Ni, Ni).

        Args:
            data (Data):
                - spatial_pos (LongTensor): block-diagonal distances
                of shape (ΣNi, ΣNi).
                - ptr (LongTensor): cumulative node counts, length B+1.

        Returns:
            List[LongTensor]:
                List of B distance matrices, each of shape (Ni, Ni).

        Raises:
            ValueError: if `spatial_pos` is missing or not 2D, or if `ptr`
                        is missing.
        """
        pos = getattr(data, "spatial_pos", None)
        ptr = getattr(data, "ptr", None)
        if pos is None or ptr is None or pos.dim() != 2:
            raise ValueError(f"Expected 2D spatial_pos and ptr, got "
                             f"{None if pos is None else tuple(pos.shape)}")
        raw = []
        for i in range(len(ptr) - 1):
            start, end = int(ptr[i]), int(ptr[i + 1])
            raw.append(pos[start:end, start:end])
        return raw

    def _embed_bias(self, distances: torch.LongTensor) -> torch.Tensor:
        """Embed distances into bias tensor.

        Args:
            distances (torch.LongTensor): Distance matrix of shape (B, L, L).

        Returns:
            torch.Tensor: Bias tensor of shape (B, L, L, H) where
                B = number of graphs,
                L = number of nodes in the graph (+1 if super-node = True),
                H = number of attention heads.
        """
        return self.spatial_embeddings(distances)
