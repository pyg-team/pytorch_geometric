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

import torch
import torch.nn as nn
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

    def _extract_raw_distances(self, data: Data) -> torch.LongTensor:
        """Build a batched distance matrix from `data.spatial_pos`.

        - If `data.spatial_pos` is 2D with `data.ptr`, splits rows
          by ptr into B graphs.
        - If 2D square, unsqueeze to batch dim.
        - If 3D, return as-is

        Args:
            data (Data): Batched graph data containing `spatial_pos`.

        Returns:
            torch.LongTensor: Distance matrix of shape (B, L, L) where
                B = number of graphs,
                L = number of nodes in the graph (+1 if super-node = True).
        """
        pos = data.spatial_pos
        if pos.dim() == 2 and hasattr(data, 'ptr'):
            ptr = data.ptr
            return torch.stack(
                [pos[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)], dim=0
            )
        if pos.dim() == 2 and pos.size(0) == pos.size(1):
            return pos.unsqueeze(0)
        if pos.dim() == 3:
            return pos
        raise ValueError(f"Unexpected spatial_pos shape {tuple(pos.shape)}")

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
