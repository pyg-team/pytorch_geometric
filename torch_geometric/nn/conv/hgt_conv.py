from typing import Union, Dict, Optional
from torch_geometric.typing import NodeType, EdgeType, Metadata

import math
from itertools import chain

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing

from ..inits import glorot, ones, reset


class HGTConv(MessagePassing):
    r"""The heterogeneous graph transformer operator from the `"Heterogeneous
    Graph Transformer" <https://arxiv.org/abs/2003.01332>`_ paper

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))

        self.a_rel = torch.nn.ParameterDict()
        self.m_rel = torch.nn.ParameterDict()
        self.p_rel = torch.nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]]  # Support both
    ) -> Dict[NodeType, Tensor]:
        """"""

        H, D = self.heads, self.out_channels // self.heads

        k_dict: Dict[str, Tensor] = {}
        q_dict: Dict[str, Tensor] = {}
        v_dict: Dict[str, Tensor] = {}
        out_dict: Dict[str, Tensor] = {}
        count_dict: Dict[str, float] = {}

        # Iterate over node-types:
        it = zip(self.k_lin.items(), self.q_lin.values(), self.v_lin.values())
        for (node_type, k_lin), q_lin, v_lin in it:
            x = x_dict[node_type]
            k_dict[node_type] = k_lin(x).view(-1, H, D)
            q_dict[node_type] = q_lin(x).view(-1, H, D)
            v_dict[node_type] = v_lin(x).view(-1, H, D)
            count_dict[node_type] = 0.

        # Iterate over edge-types:
        it = zip(self.a_rel.items(), self.m_rel.values(), self.p_rel.values())
        for (edge_type, a_rel), m_rel, p_rel in it:
            src_type, rel_type, dst_type = edge_type.split('__')
            edge_index = edge_index_dict[(src_type, rel_type, dst_type)]

            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
            out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                 rel=p_rel, size=None)

            if dst_type in out_dict:
                out_dict[dst_type] = out_dict[dst_type] + out
            else:
                out_dict[dst_type] = out
            count_dict[dst_type] += 1.

        # Iterate over node-types:
        it = zip(self.a_lin.items(), self.skip.values())
        for (node_type, a_lin), skip in it:
            out = out_dict[node_type]
            out = out / max(count_dict[node_type], 1.0)  # Mean aggregation.
            out = F.gelu(out)
            out = a_lin(out)
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = skip.sigmoid()
                out = alpha * alpha + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')
