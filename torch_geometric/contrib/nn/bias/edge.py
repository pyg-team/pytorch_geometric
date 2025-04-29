"""Learnable edge-based bias provider for graph attention.
It adds learnable biases to the attention logits based on edge types
between nodes, supporting both simple edges and multi-hop connections.

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


class GraphAttnEdgeBias(BaseBiasProvider):
    """Learnable edge‐feature bias provider for graph attention.

    Maps edge types in `data.edge_dist` to per‑head additive biases.
    Supports single and batched graphs via `ptr`.

    Args:
        num_heads (int): Number of attention heads.
        num_edges (int): Number of edge types (excl. padding).
        edge_type (str): 'simple' or 'multi_hop'.
        multi_hop_max_dist (int, optional): max distance for multi-hop.
        use_super_node (bool): Reserve extra index for super‑node.
    """

    def __init__(
        self,
        num_heads: int,
        num_edges: int,
        edge_type: str = 'simple',
        multi_hop_max_dist: int = None,
        use_super_node: bool = False,
    ):
        super().__init__(num_heads, use_super_node)
        num_tokens = num_edges + 1 + (1 if use_super_node else 0)
        self.super_idx = num_tokens - 1 if use_super_node else None
        self.edge_embeddings = nn.Embedding(num_tokens, num_heads)
        init.xavier_uniform_(self.edge_embeddings.weight)
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist

    def _extract_raw_distances(self, data: Data) -> torch.LongTensor:
        """Build a batched edge distance matrix from `data.edge_dist`.
        - If `data.edge_dist` is 2D with `data.ptr`, splits rows
          by ptr into B graphs.
        - If 2D square, unsqueeze to batch dim.
        - If 3D, return as-is
        Args:
            data (Data): Graph data containing `edge_dist` and optional `ptr`.

        Returns:
            torch.LongTensor: Edge distance matrix of shape
                (B, L, L) where B = number of graphs,
                L = number of nodes in the graph (+1 if super-node = True).
        """
        pos = getattr(data, 'edge_dist', None)
        if pos is None:
            B = len(data.ptr) - 1
            L = data.x.size(0) // B
            return data.x.new_zeros(B, L, L, dtype=torch.long)
        if pos.dim() == 2 and hasattr(data, 'ptr'):
            ptr = data.ptr
            return torch.stack(
                [pos[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)], dim=0
            )
        if pos.dim() == 2 and pos.size(0) == pos.size(1):
            return pos.unsqueeze(0)
        if pos.dim() == 3:
            return pos
        raise ValueError(f"Unexpected edge_dist shape {tuple(pos.shape)}")

    def _embed_bias(self, distances: torch.LongTensor) -> torch.Tensor:
        """Embed edge distances into bias tensor.

        Args:
            distances (torch.LongTensor): Edge distance matrix of shape
                (B, L, L) where B = number of graphs,
                L = number of nodes in the graph (+1 if super-node = True).

        Returns:
            torch.Tensor: Bias tensor of shape (B, L, L, H) where
                B = number of graphs,
                L = number of nodes in the graph (+1 if super-node = True),
                H = number of attention heads.
        """
        if (self.edge_type == 'multi_hop'
                and self.multi_hop_max_dist is not None):
            distances = distances.clamp(max=self.multi_hop_max_dist)
        return self.edge_embeddings(distances)
