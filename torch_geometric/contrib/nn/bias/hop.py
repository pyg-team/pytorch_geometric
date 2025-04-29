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

from torch_geometric.contrib.nn.bias.base import BaseBiasProvider
from torch_geometric.data import Data


class GraphAttnHopBias(BaseBiasProvider):
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
        super().__init__(num_heads, use_super_node)
        total_tokens = num_hops + (1 if use_super_node else 0)
        self.super_idx = total_tokens - 1 if use_super_node else None
        self.hop_embeddings = nn.Embedding(total_tokens, num_heads)
        init.xavier_uniform_(self.hop_embeddings.weight)

    def _extract_raw_distances(self, data: Data) -> torch.LongTensor:
        pos = data.hop_dist
        if pos.dim() == 2 and hasattr(data, 'ptr'):
            ptr = data.ptr
            return torch.stack(
                [pos[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)], dim=0
            )
        if pos.dim() == 2 and pos.size(0) == pos.size(1):
            return pos.unsqueeze(0)
        if pos.dim() == 3:
            return pos
        raise ValueError(f"Unexpected hop_dist shape {tuple(pos.shape)}")

    def _embed_bias(self, distances: torch.LongTensor) -> torch.Tensor:
        return self.hop_embeddings(distances)
