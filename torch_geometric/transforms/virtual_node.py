import copy

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('virtual_node')
class VirtualNode(BaseTransform):
    r"""Appends a virtual node to the given homogeneous graph that is connected
    to all other nodes, as described in the `"Neural Message Passing for
    Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper
    (functional name: :obj:`virtual_node`).
    The virtual node serves as a global scratch space that each node both reads
    from and writes to in every step of message passing.
    This allows information to travel long distances during the propagation
    phase.

    Node and edge features of the virtual node are added as zero-filled input
    features.
    Furthermore, special edge types will be added both for in-coming and
    out-going information to and from the virtual node.
    """
    def forward(self, data: Data) -> Data:
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))

        arange = torch.arange(num_nodes, device=row.device)
        full = row.new_full((num_nodes, ), num_nodes)
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_nodes, ), int(edge_type.max()) + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)

                if key == 'edge_weight':
                    size = list(value.size())
                    size[dim] = 2 * num_nodes
                    data[key] = torch.cat(
                        [value, torch.ones(size, device=value.device)],
                        dim=dim)
                elif key == 'batch':
                    data[key] = torch.cat(
                        [value,
                         torch.tensor([value[0]], device=value.device)],
                        dim=dim)
                elif data.is_edge_attr(key):
                    size = list(value.size())
                    size[dim] = 2 * num_nodes
                    data[key] = torch.cat(
                        [value, torch.zeros(size, device=value.device)],
                        dim=dim)
                elif data.is_node_attr(key):
                    size = list(value.size())
                    size[dim] = 1
                    data[key] = torch.cat(
                        [value, torch.zeros(size, device=value.device)],
                        dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes + 1

        return data
