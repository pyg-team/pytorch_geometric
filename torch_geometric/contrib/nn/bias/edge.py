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

from torch_geometric.data import Data


class GraphAttnEdgeBias(nn.Module):
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
        super().__init__()
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.use_super_node = use_super_node

        # reserve extra idx for padding (0) and optional super-node
        num_embeddings = num_edges + 1 + (1 if use_super_node else 0)
        self.edge_embeddings = nn.Embedding(num_embeddings, num_heads)
        init.xavier_uniform_(self.edge_embeddings.weight)

    def forward(self, data: Data) -> torch.Tensor:
        pos = self._prepare_edge_dist(data)
        if (self.edge_type == 'multi_hop'
                and self.multi_hop_max_dist is not None):
            pos = pos.clamp(max=self.multi_hop_max_dist)

        if self.use_super_node:
            pos = self._inject_super_node_index(pos)

        # (B, L, L) -> (B, L, L, H)
        bias = self.edge_embeddings(pos)
        return bias.permute(0, 3, 1, 2)

    def _prepare_edge_dist(self, data: Data) -> torch.LongTensor:
        if not hasattr(data, "edge_dist"):
            B = data.ptr.numel() - 1
            L = data.x.size(0) // B
            return data.x.new_zeros(B, L, L, dtype=torch.long)

        pos = data.edge_dist
        # Single graph: (L, L)
        if pos.dim() == 2 and pos.size(0) == pos.size(1):
            return pos.unsqueeze(0)
        # Batched 2D: (N, L) rows cat'd by ptr
        if pos.dim() == 2:
            if not hasattr(data, "ptr"):
                raise AttributeError(
                    "Batched `Data` needs `ptr` to split edge_dist."
                )
            ptr = data.ptr
            B = ptr.numel() - 1
            graphs = [pos[ptr[i]:ptr[i + 1]] for i in range(B)]
            return torch.stack(graphs, dim=0)
        # Already batched: (B, L, L)
        if pos.dim() == 3:
            return pos
        raise ValueError(
            f"`edge_dist` must be 2D or 3D, got {tuple(pos.shape)}"
        )

    def _inject_super_node_index(
        self, pos: torch.LongTensor
    ) -> torch.LongTensor:
        mask = torch.zeros_like(pos, dtype=torch.bool)
        mask[:, 0, :] = True
        mask[:, :, 0] = True
        super_idx = self.edge_embeddings.num_embeddings - 1
        return torch.where(mask, super_idx, pos)
