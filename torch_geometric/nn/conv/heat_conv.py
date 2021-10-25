from typing import Optional
from torch_geometric.typing import Adj, Size, OptTensor

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear, HeteroLinear


class HEATConv(MessagePassing):
    '''
    1. type-specific transformation for nodes of different types.
       transform nodes from different vector space to the same vector space.
    2. edges are assumed to have different types but contains the same kind of attributes.
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 num_node_types: int, num_edge_types: int,
                 edge_type_emb_dim: int, edge_dim: int, edge_attr_emb_dim: int,
                 heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 root_weight: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.root_weight = root_weight

        self.hetero_lin = HeteroLinear(in_channels, out_channels,
                                       num_node_types, bias=bias)

        self.edge_type_emb = Embedding(num_edge_types, edge_type_emb_dim)
        self.edge_attr_emb = Linear(edge_dim, edge_attr_emb_dim, bias=False)

        self.att = Linear(
            2 * out_channels + edge_type_emb_dim + edge_attr_emb_dim,
            self.heads, bias=False)

        self.lin = Linear(out_channels + edge_attr_emb_dim, out_channels,
                          bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.hetero_lin.reset_parameters()
        self.edge_type_emb.reset_parameters()
        self.edge_attr_emb.reset_parameters()
        self.att.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, node_type: Tensor,
                edge_type: Tensor, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        x = self.hetero_lin(x, node_type)

        edge_type_emb = F.leaky_relu(self.edge_type_emb(edge_type),
                                     self.negative_slope)

        # propagate_type: (x: Tensor, edge_type_emb: Tensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, edge_type_emb=edge_type_emb,
                             edge_attr=edge_attr, size=size)

        if self.concat:
            if self.root_weight:
                out += x.view(-1, 1, self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            if self.root_weight:
                out += x

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type_emb: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = F.leaky_relu(self.edge_attr_emb(edge_attr),
                                 self.negative_slope)

        alpha = torch.cat([x_i, x_j, edge_type_emb, edge_attr], dim=-1)
        alpha = F.leaky_relu(self.att(alpha), self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin(torch.cat([x_i, edge_attr], dim=-1)).unsqueeze(-2)
        out = out * alpha.unsqueeze(-1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
