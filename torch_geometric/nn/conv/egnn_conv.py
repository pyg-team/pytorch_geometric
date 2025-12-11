from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    Size,
)

import torch_scatter

eps = 1e-8

class EGNNConv(MessagePassing):
    """
    Equivariant Graph Neural Network Layer.
    Args:
        nn_edge  (Callable): Neural network for edge message computation.
        nn_node  (Callable): Neural network for node feature updates.
        pos_dim  (int): Dimension of node positions. 
        nn_pos   (Callable): Neural network for position updates. (optional)
        skip_connection (bool): If set to :obj:`True`, adds skip connections to node features. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **x**: node features :math:`(|\mathcal{V}|, F_{in})`
        - **pos**: node positions :math:`(|\mathcal{V}|, pos_{dim})`
        - **edge_index**: graph connectivity :math:`(2, |\mathcal{E}|)`
        - **edge_attr**: edge features :math:`(|\mathcal{E}|, F_{edge})` *(optional)*
        - **output**: tuple containing
            - updated node features :math:`(|\mathcal{V}|, F_{out})`
            - updated node positions :math:`(|\mathcal{V}|, pos_{dim})`
        """
    def __init__(self, 
                 nn_edge        : Callable,
                 nn_node        : Callable,
                 pos_dim        : int,
                 nn_pos         : Optional[Callable] = None,
                 skip_connection: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.nn_edge = nn_edge
        self.nn_node = nn_node
        self.nn_pos = nn_pos
        self.skip_connection = skip_connection
        self.pos_dim = pos_dim
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        for nn in [self.nn_edge, self.nn_node]:
            reset(nn)
        if self.nn_pos is not None:
            reset(self.nn_pos)

    def forward(self,
                x          : Tensor,
                pos        : Tensor,
                edge_index : Adj, 
                edge_attr  : Optional[Tensor] = None,
                size       : Size = None) -> Tuple[Tensor, Tensor]:

        assert x.size(0) == pos.size(0), "x and pos must have same number of nodes"
        if edge_attr is not None:
            assert edge_attr.size(0) == edge_index.size(1), "edge_attr must match number of edges"
        assert self.skip_connection is False or x.size(-1) == self.nn_node[-1].out_features, "For skip connection, input and output node feature dimensions must match"
    
        # Perform message passing.
        message = self.propagate(edge_index, x=(x, x), pos=(pos, pos), edge_attr=edge_attr, size=size)
        if self.nn_pos is not None:
            (message_node, message_pos) = torch.split(message, [message.size(-1) - self.pos_dim, self.pos_dim], dim=-1)
            out_pos = pos + message_pos
        else:
            message_node = message
            out_pos = pos
        
        # Update node features.
        out_node = self.nn_node(torch.cat([x, message_node], dim=-1))
        if self.skip_connection:
            out_node = out_node + x
        return (out_node, out_pos)

    def message(self, 
                x_i      : Tensor, 
                x_j      : Tensor, 
                pos_i    : Tensor, 
                pos_j    : Tensor, 
                edge_attr: Optional[Tensor] = None) -> Tensor:

        pos_diff = pos_i - pos_j
        square_norm = torch.sum(pos_diff ** 2, dim=-1).unsqueeze(-1)
        if edge_attr is not None:
            out = torch.cat([x_j, x_i, square_norm, edge_attr], dim=-1)
        else:
            out = torch.cat([x_j, x_i, square_norm], dim=-1)

        out = self.nn_edge(out)

        if self.nn_pos is not None:
            out_pos_j = self.nn_pos(out)
            pos_diff = pos_diff/(torch.sqrt(square_norm + eps) + eps)
            message_pos_j = pos_diff * out_pos_j
            out = torch.cat([out, message_pos_j], dim=-1)
 
        return out
    
    def aggregate(self, 
                  inputs  : Tensor, 
                  index   : Tensor, 
                  dim_size: Optional[int] = None) -> Tensor:
        
        out_pos = None
        if self.nn_pos is not None:
            (message_node, message_pos) = torch.split(inputs, [inputs.size(-1) - self.pos_dim, self.pos_dim], dim=-1)
            out_pos = torch_scatter.scatter_mean(message_pos, index, dim=0, dim_size=dim_size)
        else:
            message_node = inputs
        
        out = torch_scatter.scatter_add(message_node, index, dim=0, dim_size=dim_size)

        if out_pos is not None:
            out = torch.cat([out, out_pos], dim=-1)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn_edge={self.nn_edge}, nn_node={self.nn_node}, nn_pos={self.nn_pos if self.nn_pos is not None else "None"})'
    