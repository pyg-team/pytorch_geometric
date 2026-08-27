import copy

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import ClusterData, Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


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
    recursive: only used if clustering is set to True.If set to :obj:`True`, will use multilevel recursive bisection instead of multilevel k-way partitioning. default = False

    modified from VirtualNode class
    """
    def __init__(self, n_to_add: int, clustering: bool = False,
                 recursive: bool = False) -> None:
        self.n = n_to_add
        self.clustering = clustering
        self.recursive = recursive

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
            # run clustering algorithm to assign virtual node i to nodes in cluster i
            clustered_data = ClusterData(data, self.n,
                                         recursive=self.recursive)
            partition = clustered_data.partition
            assignments = []
            for i in range(self.n):
                # get nodes in cluster i
                start = int(partition.partptr[i])
                end = int(partition.partptr[i + 1])
                cluster_nodes = partition.node_perm[start:end]
                if len(cluster_nodes) == 0:
                    print(
                        f"Cluster {i + 1} is empty after running the METIS algorithm with recursion set to {self.recursive}. Decreasing number of virtual nodes added to {self.n - 1}."
                    )
                    self.n -= 1  # decrease the number of virtual nodes to add
                    continue
                assignments.append(np.array(cluster_nodes))

        arange = torch.from_numpy(np.concatenate(assignments))
        # accounts for uneven splitting
        full = torch.cat([
            torch.full(
                (len(assignments[i]), ), num_nodes + i, device=row.device)
            for i in range(self.n)
        ], dim=-1)

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
