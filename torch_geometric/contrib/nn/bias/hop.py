"""Learnable hop-based bias provider for graph attention.
It adds learnable biases to the attention logits based on multi-hop
connectivity between nodes, where different hop distances receive
separate bias terms for each attention head.

Ref:

Yunsheng Bai, Lili Wang, ”Transformer for Graphs: An Overview from
Architecture Perspective,” 2022. arXiv preprint arXiv:2202.08455,


Code adapted from:
https://github.com/qwerfdsaplking/Graph-Trans
"""

import torch
import torch.nn as nn
from torch.nn import init

from torch_geometric.data import Data


class GraphAttnHopBias(nn.Module):
    """Learnable hop‐distance bias provider for graph attention.

    Maps integer hop distances in `data.hop_dist` to per‑head additive
    biases. Supports single and batched graphs via `ptr`.

    Args:
        num_heads (int): Number of attention heads.
        num_hops (int): Max hop distance (excl. super node).
        use_super_node (bool): Reserve an extra index for super‑node.
    """

    def __init__(
        self, num_heads: int, num_hops: int, use_super_node: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_super_node = use_super_node
        # Reserve extra index for super node if needed
        self.num_hops = num_hops + (1 if use_super_node else 0)

        self.hop_embeddings = nn.Embedding(self.num_hops, num_heads)
        init.xavier_uniform_(self.hop_embeddings.weight)

    def forward(self, data: Data) -> torch.Tensor:
        pos = self._prepare_hop_dist(data)
        if self.use_super_node:
            pos = self._inject_super_node_index(pos)

        # (B, L, L) -> (B, L, L, H)
        bias = self.hop_embeddings(pos)
        return bias.permute(0, 3, 1, 2)

    def _prepare_hop_dist(self, data: Data) -> torch.LongTensor:
        if not hasattr(data, "hop_dist"):
            raise AttributeError("`Data` requires a `hop_dist` field.")
        pos = data.hop_dist
        # Single graph: (L, L)
        if pos.dim() == 2 and pos.size(0) == pos.size(1):
            return pos.unsqueeze(0)
        # Batched 2D: (N, L) rows cat’d by ptr
        if pos.dim() == 2:
            if not hasattr(data, "ptr"):
                raise AttributeError(
                    "Batched `Data` needs `ptr` to split hop_dist."
                )
            ptr = data.ptr
            B = ptr.numel() - 1
            graphs = [pos[ptr[i]:ptr[i + 1]] for i in range(B)]
            return torch.stack(graphs, dim=0)
        # Already batched: (B, L, L)
        if pos.dim() == 3:
            return pos
        raise ValueError(
            f"`hop_dist` must be 2D or 3D, got {tuple(pos.shape)}"
        )

    def _inject_super_node_index(
        self, pos: torch.LongTensor
    ) -> torch.LongTensor:
        B, L, _ = pos.shape
        mask = torch.zeros_like(pos, dtype=torch.bool)
        mask[:, 0, :] = True
        mask[:, :, 0] = True
        super_idx = self.num_hops - 1
        return torch.where(mask, super_idx, pos)
