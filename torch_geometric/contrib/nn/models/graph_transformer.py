import torch
import torch.nn as nn
from torch.nn import ModuleList

from torch_geometric.nn import global_mean_pool


class IdentityLayer(nn.Module):
    """Custom identity layer that accepts both x and batch arguments
    but only returns x.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        """Identity operation that ignores batch."""
        return x


class EncoderLayers(nn.Module):
    """Container for transformer encoder layers."""
    def __init__(self, num_layers: int = 0):
        super().__init__()
        self.layers = ModuleList([IdentityLayer() for _ in range(num_layers)])

    def forward(self, x, batch):
        for layer in self.layers:
            x = layer(x, batch)
        return x


class GraphTransformer(torch.nn.Module):
    r"""The graph transformer model from the "Transformer for Graphs:
    An Overview from Architecture Perspective"
    <https://arxiv.org/pdf/2202.08455>_ paper.
    """
    def __init__(
        self,
        hidden_dim: int = 16,
        num_class=2,
        use_super_node: bool = False,
        node_feature_encoder=nn.Identity(),
        num_encoder_layers: int = 0,
    ) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_class)
        self.use_super_node = use_super_node
        if self.use_super_node:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.node_feature_encoder = node_feature_encoder
        self.encoder = EncoderLayers(num_encoder_layers)

    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Aggregates node features into graph-level features.
        Uses the cls token readout if `use_super_node` is True,
        otherwise applies a global mean pooling.

        Args:
            x (torch.Tensor): Node feature tensor.
            batch (torch.Tensor): Batch vector mapping node to graph.

        Returns:
            torch.Tensor: Graph-level feature (num_graphs, hidden_dim).
        """
        if self.use_super_node:
            x_with_cls = self._add_cls_token(x, batch)
            return x_with_cls[:, 0, :]
        else:
            return global_mean_pool(x, batch)

    def _add_cls_token(self, x: torch.Tensor,
                       batch: torch.Tensor) -> torch.Tensor:
        """Prepends the learnable class token to every graph's node embeddings.

        Args:
            x (torch.Tensor): The input features.
            batch (torch.Tensor): The batch vector.

        Returns:
            torch.Tensor: A stacked tensor of shape
            (num_graphs, num_nodes_i+1, hidden_dim) where
            the first token of every graph corresponds to the cls token.
        """
        num_graphs = batch.max().item() + 1
        x_list = []
        for i in range(num_graphs):
            mask = batch == i
            x_i = x[mask]
            x_i = torch.cat([self.cls_token.expand(1, -1), x_i], dim=0)
            x_list.append(x_i)
        return torch.stack(x_list, dim=0)

    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes node features using the node feature encoder.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The encoded node features.
        """
        if self.node_feature_encoder is not None:
            x = self.node_feature_encoder(x)
        return x

    def forward(self, data):
        r"""Applies the graph transformer model to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            torch.Tensor: The output of the model.
        """
        x = data.x
        x = self._encode_nodes(x)
        x = self.encoder(x, data.batch)
        x = self._readout(x, data.batch)
        logits = self.classifier(x)
        return {
            "logits": logits,
        }

    def __repr__(self):
        return "GraphTransformer()"
