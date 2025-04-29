import torch
import torch.nn as nn

from torch_geometric.data import Data

from .base import BasePositionalEncoder


class SVDEncoder(BasePositionalEncoder):
    """SVD-based positional encoder.

    Takes concatenated left and right singular vectors [U, V] and projects
    them to hidden dimension with sign flip augmentation.

    Args:
        r (int): Number of singular vectors (half the input dimension)
        hidden_dim (int): Output dimension
        seed (int, optional): Random seed for sign flips. Defaults to None.
    """

    def __init__(self, r: int, hidden_dim: int, seed: int = None):
        super().__init__()
        self.r = r
        if seed is not None:
            torch.manual_seed(seed)
        self.proj = nn.Linear(2 * r, hidden_dim)
        self.sign = 2 * torch.randint(0, 2, (1, )) - 1

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            data (Data): Graph data object containing svd_pos_emb [N, 2*r]

        Returns:
            torch.Tensor: Node embeddings of shape [N, hidden_dim]
        """
        pos = data.svd_pos_emb
        if pos.size(1) != 2 * self.r:
            raise ValueError(
                f"Expected svd_pos_emb with {2*self.r} features,\
                    got {pos.size(1)}"
            )

        u, v = pos[:, :self.r], pos[:, self.r:]
        return self.proj(torch.cat([u * self.sign, v * (-self.sign)], dim=-1))
