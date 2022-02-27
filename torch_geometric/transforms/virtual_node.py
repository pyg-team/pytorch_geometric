from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops


class VirtualNode(BaseTransform):
    r"""Appends a virtual node to the given homogeneous graph which is
    connected to all other nodes in the graph as first described in the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    The virtual node serves as a global scratch space that each node both reads
    from and writes to in every step of message passing.
    This allows information to travel long distances during the propagation
    phase.

    Node and edge features of the virtual node are added as zero-filled input
    features.
    Furthermore, special edge types will be added both for in-coming and
    out-going information to and from the virtual node.
    """
    def __call__(self, data: Data) -> Data:
        N, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))

        arange = torch.arange(N, device=row.device)
        full = torch.full((N, ), N, dtype=row.dtype, device=row.device)
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        C = int(edge_type.max()) + 1
        edge_type_1 = torch.full((N, ), C, dtype=row.dtype, device=row.device)
        edge_type_2 = edge_type_1 + 1
        edge_type = torch.cat([edge_type, edge_type_1, edge_type_2], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if data.is_edge_attr(key):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())
                size[dim] = 2 * data.num_nodes
                data[key] = torch.cat([value, value.new_zeros(size)], dim=dim)

            elif data.is_node_attr(key):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())
                size[dim] = 1
                data[key] = torch.cat([value, value.new_zeros(size)], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        return data
