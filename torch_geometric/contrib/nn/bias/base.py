from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from torch_geometric.data import Data


class BaseBiasProvider(nn.Module):
    """Base class for graph attention bias providers.

    Validates tensor shapes and handles optional super-node padding.

    Args:
        num_heads (int): Number of attention heads.
        use_super_node (bool): If True, reserve one extra index for
            any attention to/from a “super” node at row/col 0.

    Abstract Methods:
        _extract_raw_distances(data: Data) -> torch.LongTensor:
            Extract node-to-node distance matrix from graph data.
        _embed_bias(distances: torch.LongTensor) -> torch.Tensor:
            Convert distance indices to bias embeddings.

    Attributes:
        num_heads (int): Number of attention heads.
        use_super_node (bool): Whether to reserve an extra index for
            the super node.
    """

    def __init__(
        self,
        num_heads: int,
        use_super_node: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.use_super_node = use_super_node
        super().__init__()

    def forward(self, data: Data) -> torch.Tensor:
        """Compute additive attention bias for a batch of graphs.

        Args:
            data (Data): Batched graph data containing fields
                required by subclasses (_extract_raw_distances).

        Returns:
            torch.Tensor: Bias of shape (B, H, L, L) where
                B = number of graphs,
                H = number of attention heads,
                L = number of nodes in the graph (+1 if super-node = True).
        """
        distances = self._extract_raw_distances(data)
        distances_padded = self._pad_to_tensor(distances)
        self._assert_square(distances_padded)

        if self.use_super_node:
            distances_padded = self._pad_with_super_node(distances_padded)

        bias = self._embed_bias(distances_padded)
        self._assert_embed_shape(bias)

        return bias.permute(0, 3, 1, 2)

    @abstractmethod
    def _extract_raw_distances(self, data: Data) -> torch.LongTensor:
        """Extract node-to-node distance matrix.

        Subclasses must produce a LongTensor of shape (B, L, L),
        where B is batch size and L is nodes per graph.
        This raw matrix will then be padded if use_super_node
        """
        ...

    @abstractmethod
    def _embed_bias(self, distances: torch.LongTensor) -> torch.Tensor:
        """Convert distance indices to bias embeddings.

        Args:
            distances (LongTensor): Indices of shape
                (B, L, L) or (B, L+1, L+1).

        Returns:
            Tensor: Embeddings of shape
                (B, L, L, H) or (B, L+1, L+1, H).
        """
        ...

    def _pad_with_super_node(
        self, distances: torch.LongTensor
    ) -> torch.LongTensor:
        """Insert a super-node index at row/col 0.

        Args:
            distances (LongTensor): Indices of shape
                (B, L, L) or (B, L+1, L+1).

        Returns:
            LongTensor: Padded indices of shape
                (B, L+1, L+1) if use_super_node is True.

        """
        B, L, _ = distances.shape
        padded = distances.new_full((B, L + 1, L + 1), self.super_idx)
        padded[:, 1:, 1:] = distances
        return padded

    def _assert_square(self, tensor: torch.LongTensor) -> None:
        """Validate tensor is 3D and square in trailing dims."""
        if tensor.dim() != 3 or tensor.size(1) != tensor.size(2):
            raise ValueError(
                f"Expected square 3D tensor, got {tuple(tensor.shape)}"
            )

    def _assert_embed_shape(self, bias: torch.Tensor) -> None:
        """Ensure bias embeddings have proper head dimension."""
        if bias.dim() != 4 or bias.size(-1) != self.num_heads:
            raise ValueError(
                f"Expected bias shape (*,*,*,{self.num_heads}),"
                f" got {tuple(bias.shape)}"
            )

    def _pad_to_tensor(self, matrices: List[Tensor]) -> Tensor:
        """Pad a list of square matrices to a common size and stack them.

        Given a list of square tensors of varying sizes [(N₁×N₁), (N₂×N₂), …],
        zero-pad each on the bottom and right so that they all become (L×L),
        where L = max(Nᵢ).  Finally, stack them along a new batch dimension.

        Args:
            matrices (List[Tensor]):
                List of 2D square tensors, each of shape (Nᵢ, Nᵢ).

        Returns:
            Tensor:
                A 3D tensor of shape (B, L, L), where B = len(matrices) and
                L = max_i Nᵢ, containing the zero-padded inputs.
        """
        sizes = [m.size(0) for m in matrices]
        max_n = max(sizes)
        padded = []
        for m in matrices:
            pad_right = max_n - m.size(1)
            pad_bottom = max_n - m.size(0)
            padded.append(F.pad(m, (0, pad_right, 0, pad_bottom), value=0))
        return torch.stack(padded, dim=0)
