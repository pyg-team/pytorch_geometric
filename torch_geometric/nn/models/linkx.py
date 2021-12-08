from torch_geometric.typing import OptTensor, Adj

import math

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Parameter
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn import inits
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import MessagePassing


class SparseLinear(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.kaiming_uniform(self.weight, fan=self.in_channels,
                              a=math.sqrt(5))
        inits.uniform(self.in_channels, self.bias)

    def forward(self, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (weight: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, weight=self.weight,
                             edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, weight_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return weight_j
        else:
            return edge_weight.view(-1, 1) * weight_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              weight: Tensor) -> Tensor:
        return matmul(adj_t, weight, reduce=self.aggr)


class LINKX(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, num_edge_layers: int = 1,
                 num_node_layers: int = 1, dropout: float = 0.5):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)
        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0., relu_first=True)

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=0., relu_first=True)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout, relu_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_lin.reset_parameters()
        if self.num_edge_layers > 1:
            self.edge_norm.reset_parameters()
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()
        self.final_mlp.reset_parameters()

    def forward(self, x: OptTensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        h1 = self.edge_lin(edge_index, edge_weight)
        if self.num_edge_layers > 1:
            h1 = h1.relu_()
            h1 = self.edge_norm(h1)
            self.edge_mlp(h1)

        if x is None:
            return h1

        h2 = self.node_mlp(x)
        out = self.cat_lin1(h1)
        out += self.cat_lin2(h2)
        out += h1
        out += h2
        out = out.relu_()
        return self.final_mlp(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')
