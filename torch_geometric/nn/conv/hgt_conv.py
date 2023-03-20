import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.data.hetero_data import combine_edge_slices
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import (
    EdgeType,
    Metadata,
    NodeType,
    SparseTensor,
    grouped_matmul_avail,
    pyg_lib,
)
from torch_geometric.utils import softmax


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
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
        use_gmm: bool = True,  # Internal use only
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        self.use_gmm = grouped_matmul_avail() and use_gmm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.src_types = [edge_type[0] for edge_type in self.edge_types]
        if self.use_gmm:  # pragma: no cover
            # grouped gemm allows us not to have to pad
            from torch_geometric.nn.dense import HeteroDictLinear
            self.k_lin = HeteroDictLinear(self.in_channels, self.out_channels,
                                          **kwargs)
            self.q_lin = HeteroDictLinear(self.in_channels, self.out_channels,
                                          **kwargs)
            self.v_lin = HeteroDictLinear(self.in_channels, self.out_channels,
                                          **kwargs)
            self.a_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                          self.node_types, **kwargs)
        else:
            from torch_geometric.nn.to_hetero_module import ToHeteroLinear
            self.max_channels = max(self.in_channels.values())
            self.k_lin = ToHeteroLinear(
                Linear(self.max_channels, self.out_channels, **kwargs),
                self.node_types)
            self.q_lin = ToHeteroLinear(
                Linear(self.max_channels, self.out_channels, **kwargs),
                self.node_types)
            self.v_lin = ToHeteroLinear(
                Linear(self.max_channels, self.out_channels, **kwargs),
                self.node_types)
            self.a_lin = ToHeteroLinear(
                Linear(self.out_channels, self.out_channels, **kwargs),
                self.node_types)
        self.skip = ParameterDict({
            node_type: Parameter(torch.Tensor(1))
            for node_type in self.node_types
        })

        self.a_rel = ParameterDict()
        self.m_rel = ParameterDict()
        self.p_rel = ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def _construct_dst_node_tensor(
            self, q_dict) -> Tuple[Tensor, Dict[NodeType, int]]:
        """Constructs a tensor of dst node features by concatenating
        features of different node types.
        Returns:
            q (Tensor): A tensor of destination node features.
            dst_offset (Dict[str, int]): A dictionary holding the offset of
                each node type in the concatenated tensor.
        """
        q_list = []
        dst_offset = {}
        offset = 0
        for node_type in self.node_types:
            q_list.append(q_dict[node_type])
            dst_offset[node_type] = offset
            offset += q_dict[node_type].size(0)
        q = torch.cat(q_list)
        return q, dst_offset

    def _construct_src_node_tensors(
            self, k_dict,
            v_dict) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs a tensor of source node features by concatenating
        a tranformed source node tensor for each edge type.

        Returns:
            k (Tensor): A tensor of source node features.
            v (Tensor): A tensor of source node features.
            src_offset (Dict[EdgeType, int]): A dictionary holding the offset
                of each source type in the concatenated tensor.
        """
        H, D = self.heads, self.out_channels // self.heads
        a_rels = [
            self.a_rel['__'.join(edge_type)] for edge_type in self.edge_types
        ]
        m_rels = [
            self.m_rel['__'.join(edge_type)] for edge_type in self.edge_types
        ]
        k_ins = []
        v_ins = []
        offset = 0
        src_offset = {}
        count = 0
        trans_ptr = [count]

        for edge_type in self.edge_types:
            src_type, _, _ = edge_type

            # Collect src node tesnors for each edge type.
            k_src = k_dict[src_type]
            v_src = v_dict[src_type]
            k_ins.append(k_src.view(-1, D))
            v_ins.append(v_src.view(-1, D))
            for _ in range(H):
                count += k_src.size(0)
                trans_ptr.append(count)

            src_offset[edge_type] = offset
            # Update offset.
            offset += k_src.size(0)

        trans_ptr = torch.tensor(trans_ptr)
        a_rel, m_rel = torch.cat(a_rels), torch.cat(m_rels)
        k_out = pyg_lib.ops.segment_matmul(torch.cat(k_ins), trans_ptr,
                                           a_rel).view(-1, H, D)
        v_out = pyg_lib.ops.segment_matmul(torch.cat(v_ins), trans_ptr,
                                           m_rel).view(-1, H, D)

        return k_out, v_out, src_offset

    def _construct_edge_index(self, edge_index_dict, src_offset, dst_offset,
                              sparse_size) -> Tuple[Tensor, Tensor]:
        """Constructs a tensor of edge indices by concatenating
        edge indices for each edge type. The edge indices are
        offseted by the offset of the source and destination nodes.

        Returns:
            edge_index (Tensor): A tensor of edge indices.
            p (Tensor): A tensor of weights for each edge type.
        """
        H = self.heads
        # List of edges and edge tensors.
        edge_index_list = []
        p_rels = []

        for edge_type in self.edge_types:
            _, _, dst_type = edge_type

            # Collect edges and edge tensors.
            indices = edge_index_dict[edge_type]
            # (TODO) Add support for SparseTensor w/o converting?
            convert = isinstance(indices, SparseTensor)
            if convert:
                # Convert to COO
                dst, src, _ = indices.coo()
                indices = torch.cat((src.view(1, -1), dst.view(1, -1)))
            num_edges = indices.size(1)
            p_rels.append(self.p_rel['__'.join(edge_type)].view(1, H).expand(
                num_edges, -1))
            # Add offset to edge indices. Offset needs to be applied as
            # the src and dst node tensors are concatenated.
            indices[0] = indices[0] + src_offset[edge_type]
            indices[1] = indices[1] + dst_offset[dst_type]
            edge_index_list.append(indices)

        # Concatenate all edges and edge tensors.
        p = torch.cat(p_rels, dim=0)
        e_idx = combine_edge_slices(edge_index_list)
        if convert:
            e_idx = SparseTensor(row=e_idx[0], col=e_idx[1],
                                 sparse_sizes=sparse_size)
        return e_idx, p

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict = {}, {}, {}
        out_dict = defaultdict(list)

        # Compute K, Q, V over node-types
        k_dict = {
            node_type: k_j.view(-1, H, D)
            for node_type, k_j in self.k_lin(x_dict).items()
        }
        q_dict = {
            node_type: q_j.view(-1, H, D)
            for node_type, q_j in self.q_lin(x_dict).items()
        }
        v_dict = {
            node_type: v_j.view(-1, H, D)
            for node_type, v_j in self.v_lin(x_dict).items()
        }

        q, dst_offset = self._construct_dst_node_tensor(q_dict)
        k, v, src_offset = self._construct_src_node_tensors(k_dict, v_dict)
        edge_index, p = self._construct_edge_index(edge_index_dict, src_offset,
                                                   dst_offset,
                                                   (k.size(0), q.size(0)))

        # propagate
        out = self.propagate(edge_index, k=k, q=q, v=v, rel=p, size=None)

        # Resconstruct output node embeddings dict.
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings.
        a_dict = self.a_lin({
            node_type: F.gelu(out)
            for node_type, out in out_dict.items() if out is not None
        })

        # Iterate over node-types:
        for node_type, out in out_dict.items():
            if out is None:
                out_dict[node_type] = None
                continue
            else:
                out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
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
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
