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

from torch_geometric.data import Data


class GraphAttnSpatialBias(nn.Module):
    """Learnable spatial bias provider for graph attention.

    Supports both single Data objects with `spatial_pos` of shape
    (L, L), and Batched Data where `spatial_pos` has been collated
    to (total_nodes, L) with `ptr` indicating graph boundaries.

    Args:
        num_heads (int): Number of attention heads.
        num_spatial (int): Number of spatial positions (excluding super node).
        use_super_node (bool): Whether to reserve a special index for
            super-node distances.
    """

    def __init__(
        self,
        num_heads: int,
        num_spatial: int,
        use_super_node: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.use_super_node = use_super_node

        # If using a super node, reserve one extra embedding slot.
        self.num_spatial = num_spatial + (1 if use_super_node else 0)

        # Embedding: spatial index → head-specific biases
        self.spatial_embeddings = nn.Embedding(self.num_spatial, num_heads)
        init.xavier_uniform_(self.spatial_embeddings.weight)

    @torch.jit.ignore
    def forward(self, data: Data) -> torch.Tensor:
        """Compute per-head spatial attention bias.

        Args:
            data (Data): Must have `spatial_pos`. Two formats supported:
                - data.spatial_pos: Tensor (L, L) on a single graph.
                - data.spatial_pos: Tensor (N, L) on a Batch with `ptr`
                of length (B+1), where each block of L rows
                corresponds to one graph.

        Returns:
            torch.Tensor: Tensor of shape (B, num_heads, L, L).
        """
        spatial_pos = data.spatial_pos

        # If collated as node-level (N, L), un-batch via ptr.
        if spatial_pos.dim() == 2:
            if not hasattr(data, "ptr"):
                raise AttributeError(
                    "Batched `Data` must have `ptr` to unbatch spatial_pos."
                )
            ptr = data.ptr
            batch_size = ptr.numel() - 1
            # Each graph has L nodes = ptr[i+1] - ptr[i]
            spatial_pos.size(1)
            graphs = [
                spatial_pos[ptr[i]:ptr[i + 1]]  # noqa: E203
                for i in range(batch_size)
            ]
            spatial_pos = torch.stack(graphs, dim=0)  # (B, L, L)

        if spatial_pos.dim() != 3:
            raise ValueError(
                f"`spatial_pos` must be 3D after unbatching, \
                    got {spatial_pos.shape}"
            )

        # Handle super-node positions by mapping row=0 or col=0 to final index
        if self.use_super_node:
            is_super = torch.zeros_like(spatial_pos, dtype=torch.bool)
            is_super[:, 0, :] = True
            is_super[:, :, 0] = True
            super_index = self.num_spatial - 1
            spatial_pos = torch.where(is_super, super_index, spatial_pos)

        # Lookup embeddings: (B, L, L) -> (B, L, L, H)
        bias = self.spatial_embeddings(spatial_pos)

        # Permute to (B, H, L, L)
        return bias.permute(0, 3, 1, 2)
