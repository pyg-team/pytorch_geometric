import copy
import numpy as np

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

import pymetis

@functional_transform('multiple_virtual_nodes')
class MultipleVirtualNodes(BaseTransform):
    r"""Appends a set of virtual node to the given homogeneous graph that is connected
    to subsets of the nodes in the original graph. The assignment of original nodes to
    virtual nodes is determined by either Randomness or Clustering, as described in the
    paper `"An Analysis of Virtual Nodes in Graph Neural Networks for Link Prediction"
    <https://openreview.net/pdf?id=dI6KBKNRp7>`_ paper
    (functional name: :obj:`multiple_virtual_nodes`).
    The virtual nodes serve as global scratch space that each node assigned to it both reads
    from and writes to in every step of message passing.

    Node and edge features of virtual nodes are added as zero-filled input
    features.
    Furthermore, special edge types will be added both for in-coming and
    out-going information to and from each virtual node. (shared between virtual nodes)

    Hyperparameters:
    n_to_add: number of virtual nodes (int). default = 1
    clustering: whether the clustering algorithm is used to assign virtual nodes (bool). default = False, means that assignment is random.

    modified from VirtualNode class
    """
    def __init__(
        self,
        n_to_add: int,
        clustering: bool
    ) -> None:
        self.n = n_to_add
        self.clustering = clustering

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        num_nodes = data.num_nodes
        assert num_nodes is not None
        # ensure we are adding at least 1 virtual node
        assert self.n > 0
        assert self.n <= num_nodes

        # Perform random assignment or clustering of virtual nodes
        num_virtual_edges = num_nodes * 2
        if not self.clustering:
            # random assignment of virtual nodes
            permute = np.random.permutation(num_nodes)
            assignments = np.array_split(permute, self.n)
        else:
            # Run METIS, as suggested in the paper. each node is assigned to 1 partition
            adjacency_list = [np.array([], dtype=int) for _ in range(num_nodes)]
            for node1, node2 in zip(row, col):
                adjacency_list[node1.item()] = np.append(adjacency_list[node1.item()], node2.item())
            # membership is a list like [1, 1, 1, 0, 1, 0,...] for self.n = 2
            n_cuts, membership = pymetis.part_graph(self.n, adjacency=adjacency_list)
            assignments = [np.where(membership == v_node)[0] for v_node in range(self.n)]

        arange = torch.from_numpy(np.concatenate(assignments))
        # accounts for uneven splitting
        full = torch.cat([torch.full((len(assignments[i]),), num_nodes + i, device=row.device) for i in range(self.n)],
                         dim=-1)

        # Update edge index
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        # add new virtual edge types
        num_edge_types = int(edge_type.max()) if edge_type.numel() > 0 else 0
        new_type = edge_type.new_full((num_nodes, ), num_edge_types + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        old_data = copy.copy(data)
        for key, value in old_data.items():
            # updating these attributes is handled above
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = old_data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = num_virtual_edges
                    fill_value = 1.
                elif key == 'batch':
                    size[dim] = self.n
                    fill_value = int(value[0])
                elif old_data.is_edge_attr(key):
                    size[dim] = num_virtual_edges
                    fill_value = 0.
                elif old_data.is_node_attr(key):
                    # add virtual node features (zeros)
                    size[dim] = self.n
                    fill_value = 0.

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes + self.n

        return data
