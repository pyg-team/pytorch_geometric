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
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.contrib.nn.bias.base import BaseBiasProvider
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset


class GraphAttnEdgeBias(BaseBiasProvider):
    """Learnable edge-feature bias provider for graph attention networks.

    Converts edge types in `data.edge_dist` into per-head additive biases,
    enabling attention modulation based on edge information. When handling
    batched graphs, the `ptr` attribute in a Data object represents the start
    and end indices of nodes for each graph. This allows proper partitioning
    of concatenated node features and ensures individual graph processing.

    Args:
        num_heads (int): Number of attention heads.
        num_edges (int): Number of edge types (excl. padding).
        edge_type (str): 'simple' or 'multi_hop'.
        multi_hop_max_dist (int, optional): Max distance for multi-hop.
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
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters of the bias provider.

        This method should reset all learnable parameters of the module.
        It is called during initialization and can be overridden by subclasses.

        Args:
            None
        Returns:
            None
        """
        reset(self.edge_embeddings.weight)

    def _extract_raw_distances(self, data: Data) -> List[Tensor]:
        """Extract per-graph edge-distance blocks from a flat edge_dist matrix.

        Args:
            data (Data):
                edge_dist (LongTensor or None):
                    If present, 2D tensor of shape (∑Ni, ∑Ni) whose
                    Ni×Ni diagonal blocks are per-graph edge distances.
                    If None, zero blocks are produced.
                ptr (LongTensor):
                    1D tensor of length B+1 holding cumulative node
                    counts [0, N1, N1+N2, …, ∑Ni].

        Returns:
            List[LongTensor]: A list of B edge-distance matrices, each of
            shape (Ni, Ni). Zero-filled if `edge_dist` was None.

        Raises:
            ValueError: If `ptr` is missing, or if `edge_dist` is present but
                        not 2D.
        """
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            raise ValueError("Missing `ptr` for batched edge_dist.")

        edge_dist = getattr(data, "edge_dist", None)
        blocks = []

        if edge_dist is None:
            for i in range(len(ptr) - 1):
                num_nodes = int(ptr[i + 1] - ptr[i])
                blocks.append(
                    data.x.new_zeros(num_nodes, num_nodes, dtype=torch.long))
            return blocks

        if edge_dist.dim() != 2:
            raise ValueError(
                f"Expected 2D edge_dist, got {tuple(edge_dist.shape)}")

        for i in range(len(ptr) - 1):
            start = int(ptr[i])
            end = int(ptr[i + 1])
            blocks.append(edge_dist[start:end, start:end])
        return blocks

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
