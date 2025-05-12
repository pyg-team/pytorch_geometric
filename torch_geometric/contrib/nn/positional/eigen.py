import torch
import torch.nn as nn

from torch_geometric.data import Data

from .base import BasePositionalEncoder


class EigEncoder(BasePositionalEncoder):
    """Positional encoder based on Laplacian eigenvector embeddings.

    Projects eigenvector embeddings into hidden space with
    sign flip augmentation.

    Args:
        num_eigvec (int): Number of eigenvectors used
        hidden_dim (int): Output dimension of embeddings
        seed (int, optional): Random seed for sign flips. Defaults to None.
    """

    def __init__(self, num_eigvec: int, hidden_dim: int, seed: int = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.proj = nn.Linear(num_eigvec, hidden_dim)
        sign = 2 * torch.randint(0, 2, (1, )) - 1
        self.register_buffer("sign", sign)

    def forward(self, data: Data) -> torch.Tensor:
        """Project eigenvector embeddings to hidden dimension.

        Args:
            data (Data): Graph data containing eig_pos_emb

        Returns:
            torch.Tensor: Positional encodings of shape [num_nodes, hidden_dim]
        """
        pos = data.eig_pos_emb
        return self.proj(pos * self.sign)
