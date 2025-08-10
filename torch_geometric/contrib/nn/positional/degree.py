import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

from .base import BasePositionalEncoder


class DegreeEncoder(BasePositionalEncoder):
    """Positional encoder that computes embeddings based on the in-degree
    and out-degree of nodes in a graph.

    Attributes:
        in_embed (nn.Embedding): Embedding layer for in-degree.
        out_embed (nn.Embedding): Embedding layer for out-degree.

    """
    def __init__(self, num_in: int, num_out: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_embed = nn.Embedding(num_in, hidden_dim, padding_idx=0)
        self.out_embed = nn.Embedding(num_out, hidden_dim, padding_idx=0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters of the positional encoder."""
        reset(self.in_embed)
        reset(self.out_embed)

    def forward(self, data: Data) -> torch.Tensor:
        """Compute positional encodings based on in-degree
        and out-degree of nodes.

        Args:
            data (Data): The input graph data object containing
            in_degree and out_degree attributes.

        Returns:
            torch.Tensor: Positional encodings aligned with data.x
        """
        in_d, out_d = data.in_degree, data.out_degree
        return self.in_embed(in_d) + self.out_embed(out_d)
