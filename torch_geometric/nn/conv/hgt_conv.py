from typing import Union, Dict, Tuple, Optional
from torch_geometric.typing import Metadata, Adj

import math
from itertools import chain

import torch
from torch import Tensor
from torch.nn import ModuleDict, Parameter
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing

from ..inits import glorot, ones


class HGTConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, metadata: Metadata, heads: int,
                 dropout: float = 0.0, **kwargs):

        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.k_lins = ModuleDict()
        self.q_lins = ModuleDict()
        self.v_lins = ModuleDict()
        self.a_lins = ModuleDict()
        self.skips = ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lins[node_type] = Linear(in_channels, out_channels)
            self.q_lins[node_type] = Linear(in_channels, out_channels)
            self.v_lins[node_type] = Linear(in_channels, out_channels)
            self.a_lins[node_type] = Linear(out_channels, out_channels)
            self.skips[node_type] = Parameter(torch.Tensor(1))

        self.pri_rel = ModuleDict()
        self.att_rel = ModuleDict()
        self.msg_rel = ModuleDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.pri_rel[edge_type] = Parameter(torch.Tensor(heads))
            self.att_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.msg_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in chain(self.k_lins.values(), self.q_lins.values(),
                         self.v_lins.values(), self.a_lins.values()):
            lin.reset_parameters()
        for param in chain(self.skips.values(), self.pri_rel.values()):
            ones(param)
        for param in chain(self.att_rel.values(), self.msg_rel.values()):
            glorot(param)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Adj],
    ) -> Dict[str, Tensor]:

        H, F = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict, count_dict = {}, {}, {}, {}, {}
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lins[node_type](x).view(-1, H, F)
            q_dict[node_type] = self.q_lins[node_type](x).view(-1, H, F)
            v_dict[node_type] = self.v_lins[node_type](x).view(-1, H, F)
            out_dict[node_type] = 0.
            count_dict[node_type] = 0.

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            att_rel = self.att_rel[edge_type]
            msg_rel = self.msg_rel[edge_type]
            k = (k_dict[src_type].transpose(0, 1) @ att_rel).transpose(1, 0)
            v = (k_dict[src_type].transpose(0, 1) @ msg_rel).transpose(1, 0)

            out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                 edge_type=edge_type, size=None)

            out_dict[dst_type] = out_dict[dst_type] + out
            count_dict[dst_type] += 1.

        for node_type, out in out_dict.items():
            out = out / count_dict[node_type]  # Mean type-wise aggregation.
            out = F.gelu(out)
            out = self.q_lins[node_type](out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = out * alpha + x_dict[node_type] * (1 - alpha)
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_type: str,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1) * self.pri_rel[edge_type]
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')
