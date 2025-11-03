from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList

from torch_geometric.contrib.nn.conv import MixupConv
from torch_geometric.nn import GCNConv, MessagePassing


class NodeMixup(torch.nn.Module):
    """The Mixup model for Node Classification from the
    `Mixup for Node and Graph Classification
    <https://dl.acm.org/doi/pdf/10.1145/3442381.3449796>`_ paper.

    .. note::
        For examples of using the Mixup for Node Classification, see
        `examples/contrib/node_mixup.py`.

    Args:
        num_layers (int): Number of message passing layers.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        conv_layer (torch.nn.Module, optional): The PyG convolutional layer
            to use (e.g., GCNConv, GATConv). Defaults to GCNConv.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    """
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        conv_layer: Optional[Module] = GCNConv,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, conv_layer, **kwargs))

        for _ in range(num_layers - 1):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, conv_layer,
                               **kwargs))

        self.lin = torch.nn.Linear(hidden_channels, self.out_channels)

    def init_conv(
        self,
        in_channels: int,
        out_channels: int,
        conv_layer: Optional[Module] = GCNConv,
        **kwargs,
    ) -> MessagePassing:
        return MixupConv(in_channels, out_channels, conv_layer, **kwargs)

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_index_b: Tensor,
        lam: float,
        id_new_value_old: Tensor,
    ) -> Tensor:
        """Forward pass for NodeMixup.

        Args:
            x (Tensor): Node feature matrix of the first input graph with
                shape :obj:`[num_nodes, in_channels]`.
            edge_index (Tensor): Graph connectivity in COO format with shape
                :obj:`[2, num_edges]`.
            edge_index_b (Tensor): Graph connectivity in COO format for
                shuffled graph.
            lam (float): Lambda for the mixup.
            id_new_value_old (Tensor): Mapping of node IDs after shuffle.

        Returns:
            Tensor: Node feature matrix of shape:
                obj:`[num_nodes,out_channels]`.
        """
        x0 = x
        x0_b = x0[id_new_value_old]
        x_mix = x0 * lam + x0_b * (1 - lam)

        for conv in self.convs[:-1]:
            x = conv(x0, edge_index, x_mix)
            x_b = conv(x0_b, edge_index_b, x_mix)
            x = F.relu(x)
            x_b = F.relu(x_b)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_b = F.dropout(x_b, p=self.dropout, training=self.training)

            x_mix = x * lam + x_b * (1 - lam)
            x_mix = F.dropout(x_mix, p=self.dropout, training=self.training)

            x0 = conv(x0, edge_index, x0)
            x0 = F.relu(x0)
            x0 = F.dropout(x0, p=self.dropout, training=self.training)
            x0_b = x0[id_new_value_old]

        # Final layer without ReLU
        x = self.convs[-1](x0, edge_index, x_mix)
        x_b = self.convs[-1](x0_b, edge_index_b, x_mix)
        x = F.relu(x)
        x_b = F.relu(x_b)

        # Mixup for final output
        x_out = x * lam + x_b * (1 - lam)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        # Linear layer for classification/logits
        return self.lin(x_out).log_softmax(dim=-1)
