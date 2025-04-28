from typing import Protocol

import torch

from torch_geometric.data import Data


class BiasProvider(Protocol):
    """Protocol for providing additive attention bias tensors.

    The bias tensor is added to the attention scores before softmax.
    Shape of returned tensor should be
    (batch_size, num_heads, seq_len, seq_len).
    """

    def __call__(self, data: Data) -> torch.Tensor:
        """Compute attention bias tensor for given graph data.

        Args:
            data (Data): Input graph data.

        Returns:
            torch.Tensor: Attention bias tensor with shape (B, H, L, L) and
                floating dtype.
        """
        ...
