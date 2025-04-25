import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool


class GraphTransformer(torch.nn.Module):
    r"""The graph transformer model from the "Transformer for Graphs:
    An Overview from Architecture Perspective"
    <https://arxiv.org/pdf/2202.08455>_ paper.
    """
    def __init__(self, hidden_dim: int = 16, num_class=2) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_class)

    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Applies a readout function to the input features.

        Readout functions aggregate node features into graph-level features.

        Args:
            x (torch.Tensor): The input features.
            batch (torch.Tensor): The batch vector.

        Returns:
            torch.Tensor: The output of the readout function.
        """
        return global_mean_pool(x, batch)

    def forward(self, data):
        r"""Applies the graph transformer model to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            torch.Tensor: The output of the model.
        """
        x = self._readout(data.x, data.batch)
        logits = self.classifier(x)
        return {
            "logits": logits,
        }

    def __repr__(self):
        return "GraphTransformer()"
