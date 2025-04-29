from abc import abstractmethod

import torch
import torch.nn as nn

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
        self._assert_square(distances)

        if self.use_super_node:
            distances = self._pad_with_super_node(distances)

        bias = self._embed_bias(distances)
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
