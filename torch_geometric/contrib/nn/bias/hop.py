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

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
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

    def _extract_raw_distances(self, data: Data) -> List[Tensor]:
        """Extract per-graph hop-distance blocks from a flat hop_dist matrix.

        Args:
            data (Data):
                hop_dist (LongTensor):
                    2D tensor of shape (∑Ni, ∑Ni), where each Ni×Ni
                    diagonal block is the hop-distance matrix for graph i.
                ptr (LongTensor):
                    1D tensor of length B+1 holding cumulative node
                    counts [0, N1, N1+N2, …, ∑Ni].

        Returns:
            List[LongTensor]: A list of B hop-distance matrices, where the
            i-th element has shape (Ni, Ni).

        Raises:
            ValueError: If `hop_dist` is missing or not 2D, or if `ptr`
                        is missing.
        """
        hop_dist = getattr(data, "hop_dist", None)
        ptr = getattr(data, "ptr", None)
        if hop_dist is None or ptr is None or hop_dist.dim() != 2:
            raise ValueError(
                f"Expected 2D data.hop_dist and data.ptr, got "
                f"{None if hop_dist is None else tuple(hop_dist.shape)}"
            )
        blocks = []
        for i in range(len(ptr) - 1):
            start = int(ptr[i])
            end = int(ptr[i + 1])
            blocks.append(hop_dist[start:end, start:end])
        return blocks

    def _embed_bias(self, distances: torch.LongTensor) -> torch.Tensor:
        return self.hop_embeddings(distances)
