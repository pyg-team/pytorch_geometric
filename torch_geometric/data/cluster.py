from typing import List

import copy
import os.path as osp

import torch
import torch.utils.data
from torch_sparse import SparseTensor, cat


class ClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (string, optional): If set, will save the partitioned data to
            the :obj:`save_dir` directory for faster re-use.
    """
    def __init__(self, data, num_parts, recursive=False, save_dir=None):
        assert (data.edge_index is not None)

        self.num_parts = num_parts
        self.recursive = recursive
        self.save_dir = save_dir

        self.process(data)

    def process(self, data):
        recursive = '_recursive' if self.recursive else ''
        filename = f'part_data_{self.num_parts}{recursive}.pt'

        path = osp.join(self.save_dir or '', filename)
        if self.save_dir is not None and osp.exists(path):
            data, partptr, perm = torch.load(path)
        else:
            data = copy.copy(data)
            num_nodes = data.num_nodes

            (row, col), edge_attr = data.edge_index, data.edge_attr
            adj = SparseTensor(row=row, col=col, value=edge_attr)
            adj, partptr, perm = adj.partition(self.num_parts, self.recursive)

            for key, item in data:
                if item.size(0) == num_nodes:
                    data[key] = item[perm]

            data.edge_index = None
            data.edge_attr = None
            data.adj = adj

            if self.save_dir is not None:
                torch.save((data, partptr, perm), path)

        self.data = data
        self.perm = perm
        self.partptr = partptr

    def __len__(self):
        return self.partptr.numel() - 1

    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start

        data = copy.copy(self.data)
        num_nodes = data.num_nodes

        for key, item in data:
            if item.size(0) == num_nodes:
                data[key] = item.narrow(0, start, length)

        data.adj = data.adj.narrow(1, start, length)

        row, col, value = data.adj.coo()
        data.adj = None
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.data}, '
                f'num_parts={self.num_parts})')


class ClusterLoader(torch.utils.data.DataLoader):
    r"""The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
    for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
    and their between-cluster links from a large-scale graph data object to
    form a mini-batch.

    Args:
        cluster_data (torch_geometric.data.ClusterData): The already
            partioned data object.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
    """
    def __init__(self, cluster_data, batch_size=1, shuffle=False, **kwargs):
        class HelperDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(cluster_data)

            def __getitem__(self, idx):
                start = int(cluster_data.partptr[idx])
                length = int(cluster_data.partptr[idx + 1]) - start

                data = copy.copy(cluster_data.data)
                num_nodes = data.num_nodes
                for key, item in data:
                    if item.size(0) == num_nodes:
                        data[key] = item.narrow(0, start, length)

                return data, idx

        def collate(batch):
            data_list = [data[0] for data in batch]
            parts: List[int] = [data[1] for data in batch]
            partptr = cluster_data.partptr

            adj = cat([data.adj for data in data_list], dim=0)

            adj = adj.t()
            adjs = []
            for part in parts:
                start = partptr[part]
                length = partptr[part + 1] - start
                adjs.append(adj.narrow(0, start, length))
            adj = cat(adjs, dim=0).t()
            row, col, value = adj.coo()

            data = cluster_data.data.__class__()
            data.num_nodes = adj.size(0)
            data.edge_index = torch.stack([row, col], dim=0)
            data.edge_attr = value

            ref = data_list[0]
            keys = ref.keys
            keys.remove('adj')

            for key in keys:
                if ref[key].size(0) != ref.adj.size(0):
                    data[key] = ref[key]
                else:
                    data[key] = torch.cat([d[key] for d in data_list],
                                          dim=ref.__cat_dim__(key, ref[key]))

            return data

        super(ClusterLoader,
              self).__init__(HelperDataset(), batch_size, shuffle,
                             collate_fn=collate, **kwargs)
