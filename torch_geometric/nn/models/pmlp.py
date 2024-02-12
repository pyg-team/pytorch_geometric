from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import SimpleConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class PMLP(torch.nn.Module):
    r"""The P(ropagational)MLP model from the `"Graph Neural Networks are
    Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"
    <https://arxiv.org/abs/2212.09034>`_ paper.
    :class:`PMLP` is identical to a standard MLP during training, but then
    adopts a GNN architecture during testing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): The number of layers.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        norm (bool, optional): If set to :obj:`False`, will not apply batch
            normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the module
            will not learn additive biases. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.,
        norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, self.bias))
        for _ in range(self.num_layers - 2):
            lin = Linear(hidden_channels, hidden_channels, self.bias)
            self.lins.append(lin)
        self.lins.append(Linear(hidden_channels, out_channels, self.bias))

        self.norm = None
        if norm:
            self.norm = torch.nn.BatchNorm1d(
                hidden_channels,
                affine=False,
                track_running_stats=False,
            )

        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight, gain=1.414)
            if self.bias:
                torch.nn.init.zeros_(lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """"""  # noqa: D419
        if not self.training and edge_index is None:
            raise ValueError(f"'edge_index' needs to be present during "
                             f"inference in '{self.__class__.__name__}'")

        for i in range(self.num_layers):
            x = x @ self.lins[i].weight.t()
            if not self.training:
                x = self.conv(x, edge_index)
            if self.bias:
                x = x + self.lins[i].bias
            if i != self.num_layers - 1:
                if self.norm is not None:
                    x = self.norm(x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class EdgeConv(MessagePassing):
    """
    A message passing layer that uses edge features for convolution.

    This layer extends PyTorch Geometric's `MessagePassing` class to perform
    convolutions by incorporating edge attributes into the message passing process.
    It aggregates messages using the mean aggregation function.

    Args:
        in_channels (int): Size of each input sample's features.
        out_channels (int): Size of each output sample's features.
        bias (bool, optional): If set to `False`, the layer will not learn an additive bias.
            Default is `True`.

    Methods:
        forward(x, edge_index, edge_attr): Performs the forward pass of the layer.
        message(x_j, edge_attr): Constructs messages to node i for each edge (j, i) in `edge_index`.
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(EdgeConv, self).__init__(aggr='mean')  # Mean aggregation.
        self.lin = Linear(in_channels + in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the EdgeConv layer.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (Tensor): Edge feature matrix with shape [num_edges, in_channels].

        Returns:
            Tensor: Updated node features with shape [num_nodes, out_channels].
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        Constructs messages for each node based on edge attributes.

        Args:
            x_j (Tensor): Incoming features for each edge with shape [num_edges, in_channels].
            edge_attr (Tensor): Edge features for each edge with shape [num_edges, in_channels].

        Returns:
            Tensor: Messages for each node with shape [num_edges, out_channels].
        """
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.lin(tmp)

    def __repr__(self) -> str:
        """
        Creates a string representation of this EdgeConv instance, showing the
        key configurations of the layer.

        Returns:
            str: A string representation including the class name and key layer
                 configurations such as input and output channels.
        """
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'bias={self.lin.bias is not None})')


class PMLP_with_EdgeAttr(torch.nn.Module):
    """
    Propagational MLP with Edge Attributes (PMLP_with_EdgeAttr) model.

    This model extends an MLP to incorporate edge attributes in its message passing mechanism,
    using EdgeConv layers for convolution operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden layer sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers in the MLP.
        edge_attr_dim (int): Dimensionality of edge attributes.
        dropout (float, optional): Dropout probability. Default is 0.0.
        norm (bool, optional): If `True`, applies batch normalization. Default is `True`.
        bias (bool, optional): If `True`, layers will learn an additive bias. Default is `True`.

    Methods:
        forward(x, edge_index, edge_attr): Performs the forward pass of the model.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        edge_attr_dim: int,
        dropout: float = 0.0,
        norm: bool = True,
        bias: bool = True,
    ):
        super(PMLP_with_EdgeAttr, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, bias))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels, bias))
        self.lins.append(Linear(hidden_channels, out_channels, bias))

        self.norm = None
        if norm:
            self.norm = BatchNorm1d(hidden_channels, affine=True, track_running_stats=True)

        self.conv = EdgeConv(hidden_channels + edge_attr_dim, hidden_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes or resets the model parameters."""
        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight, gain=torch.nn.init.calculate_gain('relu'))
            if self.bias:
                torch.nn.init.zeros_(lin.bias)
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the PMLP_with_EdgeAttr model.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (Tensor, optional): Edge feature matrix with shape [num_edges, edge_attr_dim].

        Returns:
            Tensor: Updated node features with shape [num_nodes, out_channels].
        """
        if not self.training and edge_index is None:
            raise ValueError(f"'edge_index' needs to be present during inference in '{self.__class__.__name__}'")

        for i in range(self.num_layers):
            x = x @ self.lins[i].weight.t()
            if not self.training:
                x = self.conv(x, edge_index, edge_attr)
            if self.bias:
                x = x + self.lins[i].bias
            if i != self.num_layers - 1:
                if self.norm is not None:
                    x = self.norm(x)
                x = F.tanh(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def __repr__(self) -> str:
        """
        Creates a string representation of this PMLP_with_EdgeAttr instance,
        showing the key configurations of the model.

        Returns:
            str: A string representation including the class name and key model
                 configurations such as input channels, hidden channels, output
                 channels, and the number of layers.
        """
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'num_layers={self.num_layers}, '
                f'edge_attr_dim={self.conv.in_channels - self.hidden_channels}, '
                f'dropout={self.dropout}, '
                f'norm={self.norm is not None}, '
                f'bias={self.bias})')
