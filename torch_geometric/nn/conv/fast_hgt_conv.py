import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import (
    Adj,
    EdgeType,
    Metadata,
    NodeType,
    SparseTensor,
)
from torch_geometric.utils import softmax


class FastHGTConv(MessagePassing):
    r"""See :class:`HGTConv`."""
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.src_types = [edge_type[0] for edge_type in self.edge_types]

        self.k_lin = HeteroDictLinear(self.in_channels, self.out_channels)
        self.q_lin = HeteroDictLinear(self.in_channels, self.out_channels)
        self.v_lin = HeteroDictLinear(self.in_channels, self.out_channels)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, is_sorted=True,
                                  bias=False)
        self.v_rel = HeteroLinear(dim, dim, num_types, is_sorted=True,
                                  bias=False)

        self.skip = ParameterDict({
            node_type: Parameter(torch.Tensor(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.Tensor(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.k_lin.reset_parameters()
        self.q_lin.reset_parameters()
        self.v_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self,
        k_dict: Dict[str, Tensor],
        v_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        count = 0
        cumsum = 0
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[int] = []
        offset: Dict[EdgeType] = {}
        for edge_type in self.edge_types:
            src, _, _ = edge_type

            ks.append(k_dict[src].view(-1, D))
            vs.append(v_dict[src].view(-1, D))

            N = k_dict[src].size(0)
            for _ in range(H):
                type_list.append(torch.full((N, ), count, dtype=torch.long))
                count += 1
            offset[edge_type] = cumsum
            cumsum += N

        type_vec = torch.cat(type_list, dim=0)
        k = self.k_rel(torch.cat(ks, dim=0), type_vec).view(-1, H, D)
        v = self.k_rel(torch.cat(vs, dim=0), type_vec).view(-1, H, D)

        return k, v, offset

    def _construct_edge_index(
        self,
        edge_index_dict: Dict[EdgeType, Adj],
        src_offset: Dict[EdgeType, int],
        dst_offset: [NodeType, int],
    ) -> Tuple[Adj, Tensor]:
        """Constructs a tensor of edge indices by concatenating edge indices
        for each edge type. The edge indices are increased by the offset of the
        source and destination nodes."""
        edge_indices: List[Tensor] = []
        ps: List[Tensor] = []

        for edge_type in self.edge_types:
            _, _, dst_type = edge_type

            edge_index = edge_index_dict[edge_type]

            # (TODO) Add support for SparseTensor w/o converting.
            is_sparse = isinstance(edge_index, SparseTensor)
            if is_sparse:  # Convert to COO
                dst, src, _ = edge_index.coo()
                edge_index = torch.stack([src, dst], dim=0)
            else:
                edge_index = edge_index.clone()

            p = self.p_rel['__'.join(edge_type)].expand(edge_index.size(1), -1)
            ps.append(p)

            # Add offset to edge indices:
            edge_index[0] += src_offset[edge_type]
            edge_index[1] += dst_offset[dst_type]
            edge_indices.append(edge_index)

        # Concatenate all edges and edge tensors:
        p = torch.cat(ps, dim=0)
        edge_index = torch.cat(edge_indices, dim=1)

        if is_sparse:
            edge_index = SparseTensor(row=edge_index[1], col=edge_index[0],
                                      value=p)

        return edge_index, p

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]  # Support both.
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

        # Compute K, Q, V over node types:
        k_dict = self.k_lin(x_dict)
        k_dict = {k: v.view(-1, H, D) for k, v in k_dict.items()}

        q_dict = self.q_lin(x_dict)
        q_dict = {k: v.view(-1, H, D) for k, v in q_dict.items()}

        v_dict = self.v_lin(x_dict)
        v_dict = {k: v.view(-1, H, D) for k, v in v_dict.items()}

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(k_dict, v_dict)

        edge_index, edge_attr = self._construct_edge_index(
            edge_index_dict, src_offset, dst_offset)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({k: F.gelu(v) for k, v in out_dict.items()})

        # Iterate over node types:
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

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
