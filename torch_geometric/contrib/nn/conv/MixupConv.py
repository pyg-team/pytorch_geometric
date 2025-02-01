from typing import Optional

import torch
from torch.nn import Module

from torch_geometric.nn import GCNConv, MessagePassing


class MixupConv(MessagePassing):
    r"""MixupConv is a flexible graph convolution layer that supports mixup
    logic between two sets of node features (`x1` and `x2`). It allows users
    to utilize different types of graph convolutions (e.g., GCNConv, GATConv)
    while maintaining the mixup logic, where messages from `x1` are passed to
    update `x2`.

    **Mixup Logic**:
    The MixupConv layer computes messages from one set of node features
    (`x1`) and applies them to update the second set of node features (`x2`),
    thus maintaining the mixup between the two feature sets.

    **Supported Convolution Types**:
    GCNConv
    GATConv
    Other PyG convolution types can be passed during initialization.

    Args:
        in_channels (int): Size of each input sample (number of input node
            features).
        out_channels (int): Size of each output sample (number of output node
            features).
        conv_layer (torch.nn.Module, optional): The PyG convolutional layer
            to use (e.g., GCNConv, GATConv). Defaults to GCNConv.
        aggr (str, optional): The aggregation scheme to use ("add", "mean",
            "max"). (default: :obj:`"mean"`)
        bias (bool, optional): If set to `False`, the layer will not learn an
            additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments passed to the convolution
            layer.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 conv_layer: Optional[Module] = GCNConv, aggr='mean',
                 bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layer = conv_layer(in_channels, out_channels, bias=bias,
                                     **kwargs)

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.conv_layer.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x1, edge_index, x2):
        """The forward function for the MixupConv layer.

        Args:
            x1 (Tensor): Node feature matrix of the first input graph with
                shape [num_nodes, in_channels].
            edge_index (Tensor): Graph connectivity in COO format with shape
                [2, num_edges].
            x2 (Tensor): Node feature matrix of the second input graph with
                shape [num_nodes, in_channels].

        Returns:
            Tensor: Node feature matrix of shape [num_nodes, out_channels],
                where the messages from `x1` are used to update `x2`.
        """
        h1 = self.conv_layer(x1, edge_index)
        return h1 + self.lin(x2)

    def __repr__(self):
        return '{}(conv={}, in_channels={}, out_channels={})'.format(
            self.__class__.__name__, self.conv_layer.__class__.__name__,
            self.in_channels, self.out_channels)
