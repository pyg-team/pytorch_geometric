import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class MixHopConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 p: list = [0, 1, 2], add_self_loops: bool = True,
                 bias: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p = p
        self.add_self_loops = add_self_loops

        self.weights = nn.ModuleDict(
            {str(j): Linear(in_channels, out_channels, bias=bias)
             for j in p})

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for _, v in self.weights.items():
            v.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        outputs = []
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight,
                                               x.size(self.node_dim), False,
                                               self.add_self_loops, self.flow,
                                               dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, edge_weight,
                                  x.size(self.node_dim), False,
                                  self.add_self_loops, self.flow,
                                  dtype=x.dtype)

        for j in range(max(self.p) + 1):
            if j in self.p:
                outputs.append(self.weights[str(j)](x))
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)

        return torch.cat(outputs, dim=1)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return edge_weight.view(-1,
                                1) * x_j if edge_weight is not None else x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, p={self.p})')
