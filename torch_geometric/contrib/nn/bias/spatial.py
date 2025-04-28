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

    Maps integer distances in `data.spatial_pos` to per-head additive
    biases. Supports:
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
        super().__init__()
        self.num_heads = num_heads
        self.use_super_node = use_super_node
        # Reserve an extra embedding index if using a super node
        self.num_spatial = num_spatial + (1 if use_super_node else 0)

        self.spatial_embeddings = nn.Embedding(self.num_spatial, num_heads)
        init.xavier_uniform_(self.spatial_embeddings.weight)

    @torch.jit.ignore
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass to compute spatial biases.

        Args:
            data (Data): Input data containing `spatial_pos` and optionally
                `ptr` for batched graphs.
        Returns:torch.Tensor: Spatial bias tensor of shape (B, H, L, L), where
            B is the batch size, H is the number of attention heads,
            and L is the number of nodes in each graph.

        """
        pos = self._prepare_spatial_pos(data)

        if self.use_super_node:
            pos = self._inject_super_node_index(pos)

        # Embedding lookup: (B, L, L) -> (B, L, L, H)
        bias = self.spatial_embeddings(pos)
        # Permute to (B, H, L, L)
        return bias.permute(0, 3, 1, 2)

    def _prepare_spatial_pos(self, data: Data) -> torch.LongTensor:
        """Turn data.spatial_pos into a (B, L, L) tensor.

        Args:
            data (Data): Input data containing `spatial_pos` and optionally
                `ptr` for batched graphs.

        Returns:
            torch.LongTensor: Prepared spatial positions tensor of shape
                (B, L, L) where B is the batch size and L is the number of
                nodes in each graph.
        """
        if not hasattr(data, "spatial_pos"):
            raise AttributeError("`Data` requires a `spatial_pos` field.")

        pos = data.spatial_pos
        # Single graph: (L, L)
        if pos.dim() == 2 and pos.size(0) == pos.size(1):
            return pos.unsqueeze(0)

        # Batched node-level: (N, L) with ptr
        if pos.dim() == 2:
            if not hasattr(data, "ptr"):
                raise AttributeError(
                    "Batched `Data` needs `ptr` to split spatial_pos."
                )
            ptr = data.ptr
            B = ptr.numel() - 1
            # Each segment pos[ptr[i]:ptr[i+1]] is an (L, L) block
            graphs = [
                pos[ptr[i]:ptr[i + 1]]  # noqa: E203
                for i in range(B)
            ]
            return torch.stack(graphs, dim=0)

        # Already proper batched: (B, L, L)
        if pos.dim() == 3:
            return pos

        raise ValueError(
            f"`spatial_pos` must be 2D or 3D, got shape {tuple(pos.shape)}"
        )

    def _inject_super_node_index(
        self, spatial_pos: torch.LongTensor
    ) -> torch.LongTensor:
        """Map row=0 or col=0 entries to the reserved super-node index.

        Args:
            spatial_pos (torch.LongTensor): Tensor of shape (B, L, L)
                containing spatial distances.

        Returns:
            torch.LongTensor: Updated tensor with super-node index
                injected for row=0 or col=0 entries.

        """
        mask = torch.zeros_like(spatial_pos, dtype=torch.bool)
        mask[:, 0, :] = True
        mask[:, :, 0] = True
        super_idx = self.num_spatial - 1
        return torch.where(mask, super_idx, spatial_pos)
