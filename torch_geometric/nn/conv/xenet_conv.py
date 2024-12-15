from typing import List, Optional, Union

import torch
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing


class XENetConv(MessagePassing):
    r"""Implementation of XENet convolution layer from the paper.

    "XENet: Using a new graph convolution to accelerate the timeline for
    protein design on quantum computers.

    Based on original implementation here:
    https://github.com/danielegrattarola/spektral/blob/master/spektral/ \
        layers/convolutional/xenet_conv.py"

    Args:
        in_node_channels (int): Size of input node features
        in_edge_channels (int): Size of input edge features
        stack_channels (Union[int, List[int]]): Number of channels for the
        hidden stack layers
        node_channels (int): Number of output node features
        edge_channels (int): Number of output edge features
        attention (bool, optional): Whether to use attention when aggregating
        messages. (default: True)
        node_activation(Optional[callable], optional): Activation function for
        nodes. (default: None)
        edge_activation (Optional[callable], optional): Activation function for
        edges. (default: None)
    """
    def __init__(self, in_node_channels: int, in_edge_channels: int,
                 stack_channels: Union[int, List[int]], node_channels: int,
                 edge_channels: int, attention: bool = True,
                 node_activation: Optional[callable] = None,
                 edge_activation: Optional[callable] = None, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_node_channels = in_node_channels
        self.in_edge_channels = in_edge_channels
        self.stack_channels = stack_channels if isinstance(
            stack_channels, list) else [stack_channels]
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.attention = attention

        # Node and edge activation functions
        self.node_activation = node_activation if node_activation is not None \
            else lambda x: x
        self.edge_activation = edge_activation if edge_activation is not None \
            else lambda x: x

        # Stack MLPs
        stack_input_size = 2 * in_node_channels + 2 * in_edge_channels
        self.stack_layers = nn.ModuleList()
        current_channels = stack_input_size

        for i, channels in enumerate(self.stack_channels):
            self.stack_layers.append(nn.Linear(current_channels, channels))
            if i != len(self.stack_channels) - 1:
                self.stack_layers.append(nn.ReLU())
            else:
                self.stack_layers.append(nn.PReLU())
            current_channels = channels

        # Final node and edge MLPs
        node_input_size = in_node_channels + 2 * self.stack_channels[-1]
        self.node_mlp = nn.Linear(node_input_size, node_channels)
        self.edge_mlp = nn.Linear(self.stack_channels[-1], edge_channels)

        # Attention layers
        if self.attention:
            self.att_in = nn.Sequential(nn.Linear(self.stack_channels[-1], 1),
                                        nn.Sigmoid())
            self.att_out = nn.Sequential(nn.Linear(self.stack_channels[-1], 1),
                                         nn.Sigmoid())

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        """Args
            x (Tensor): Node feature matrix of shape [num_nodes,
            in_node_channels] edge_index (Tensor): Graph connectivity matrix of
            shape [2, num_edges] edge_attr (Tensor): Edge feature matrix of
            shape [num_edges, in_edge_channels]

        Returns:
            tuple[Tensor, Tensor]: Updated node features [num_nodes,
            node_channels] and edge features [num_edges, edge_channels]
        """
        # Propagate messages
        out_dict = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                  size=(x.size(0), x.size(0)))

        # Update node features
        x_new = self.node_mlp(
            torch.cat([x, out_dict['incoming'], out_dict['outgoing']], dim=-1))
        x_new = self.node_activation(x_new)

        # Update edge features
        edge_features = out_dict['edge_features']
        edge_attr_new = self.edge_mlp(edge_features)
        edge_attr_new = self.edge_activation(edge_attr_new)

        return x_new, edge_attr_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> dict:
        """Constructs messages for each edge."""
        # Get reversed edge features by flipping edge_index
        edge_attr_rev = edge_attr[torch.arange(edge_attr.size(0) - 1, -1, -1)]

        # Concatenate all features
        stack = torch.cat([x_i, x_j, edge_attr, edge_attr_rev], dim=-1)

        # Apply stack MLPs
        for layer in self.stack_layers:
            stack = layer(stack)

        # Apply attention if needed
        if self.attention:
            att_in = self.att_in(stack)
            att_out = self.att_out(stack)
            stack_in = stack * att_in
            stack_out = stack * att_out
        else:
            stack_in = stack_out = stack

        return {
            'incoming': stack_in,
            'outgoing': stack_out,
            'edge_features': stack
        }

    def aggregate(self, inputs: dict, index: Tensor,
                  dim_size: Optional[int] = None) -> dict:
        """Aggregates messages from neighbors."""
        incoming = self.aggr_module(inputs['incoming'], index,
                                    dim_size=dim_size)
        outgoing = self.aggr_module(inputs['outgoing'], index,
                                    dim_size=dim_size)

        return {
            'incoming': incoming,
            'outgoing': outgoing,
            'edge_features': inputs['edge_features']
        }

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_node_channels={self.in_node_channels}, '
                f'in_edge_channels={self.in_edge_channels}, '
                f'stack_channels={self.stack_channels}, '
                f'node_channels={self.node_channels}, '
                f'edge_channels={self.edge_channels}, '
                f'attention={self.attention})')
