from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList

from torch_geometric.nn import GCNConv, global_mean_pool


class GraphMixup(torch.nn.Module):
    r"""The Mixup model for Graph Classification from the
    `Mixup for Node and Graph Classification
    <https://dl.acm.org/doi/pdf/10.1145/3442381.3449796>`_ paper.

    .. note::
        For examples of using Mixup for Graph Classification, see
        `examples/contrib/graph_mixup.py`.

    This model applies mixup at the graph-level embedding space. Given two
    batches of graphs, the embeddings of both batches are combined based on
    a mixup coefficient lambda. The final mixed embedding is then passed
    through a linear layer (or MLP) followed by a softmax output layer.

    Args:
        num_layers (int): Number of GNN layers.
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden dimension.
        out_channels (int): Number of classes.
        conv_layer (Module): A PyG convolution layer class (e.g. GCNConv).
        dropout (float): Dropout probability.
        readout (Callable): Readout function to aggregate node embeddings
        into a graph-level embedding. Defaults to global_mean_pool.
    """
    def __init__(self, num_layers: int, in_channels: int, hidden_channels: int,
                 out_channels: int, conv_layer: Module = GCNConv,
                 dropout: float = 0.5, readout: Callable = global_mean_pool,
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.readout = readout

        self.convs = ModuleList()
        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 1):
            self.convs.append(
                conv_layer(hidden_channels, hidden_channels, **kwargs))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor,
                x_b: Tensor, edge_index_b: Tensor, batch_b: Tensor,
                lam: float) -> Tensor:
        """Forward pass for the GraphMixup model.

        This method takes two batches of graph data (x, edge_index, batch)
        and (x_b, edge_index_b, batch_b) along with a mixup coefficient `lam`.
        It computes the graph-level embeddings for both batches using
        multiple GNN layers, mixes the embeddings using `lam`,
        and finally predicts the class probabilities using a linear
        layer followed by log-softmax.

        Args:
            x (Tensor): Node feature matrix for the first batch of graphs
                with shape [num_nodes, in_channels].
            edge_index (Tensor): Edge index for the first batch of graphs
                with shape [2, num_edges], defining the graph connectivity.
            batch (Tensor): Batch vector that assigns each node to a graph
                in the first batch. Shape: [num_nodes].
            x_b (Tensor): Node feature matrix for the second (shuffled)
                batch of graphs with shape [num_nodes, in_channels].
            edge_index_b (Tensor): Edge index for the second batch of graphs
                with shape [2, num_edges], defining the graph connectivity.
            batch_b (Tensor): Batch vector that assigns each node to a graph
                in the second batch. Shape: [num_nodes].
            lam (float): Mixup coefficient (lambda) from the Beta distribution,
                used to interpolate between the two graph embeddings.

        Returns:
            Tensor: The prediction logits for each graph in the batch.
                Shape: [num_graphs, out_channels], where out_channels
                is the number of classes.
        """
        # Compute embeddings for the first batch
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.readout(h, batch)

        # Compute embeddings for the second (shuffled) batch
        h_b = x_b
        for conv in self.convs:
            h_b = conv(h_b, edge_index_b)
            h_b = F.relu(h_b)
            h_b = F.dropout(h_b, p=self.dropout, training=self.training)
        h_b = self.readout(h_b, batch_b)

        # Mixup the graph embeddings
        h_mix = lam * h + (1 - lam) * h_b

        # Predict
        out = self.lin(h_mix)
        return out.log_softmax(dim=-1)
