import copy
import os.path as osp
import sys
from typing import Optional

import torch
import torch.utils.data

from torch_geometric.typing import SparseTensor, torch_sparse
from torch_geometric.utils import narrow, select


class ClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    .. note::
        The underlying METIS algorithm requires undirected graphs as input.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (str, optional): If set, will save the partitioned data to the
            :obj:`save_dir` directory for faster re-use. (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
    """
    def __init__(self, data, num_parts: int, recursive: bool = False,
                 save_dir: Optional[str] = None, log: bool = True):

        assert data.edge_index is not None

        self.num_parts = num_parts

        recursive_str = '_recursive' if recursive else ''
        filename = f'partition_{num_parts}{recursive_str}.pt'
        path = osp.join(save_dir or '', filename)
        if save_dir is not None and osp.exists(path):
            adj, partptr, perm = torch.load(path)
        else:
            if log:  # pragma: no cover
                print('Computing METIS partitioning...', file=sys.stderr)

            N, E = data.num_nodes, data.num_edges
            adj = SparseTensor(
                row=data.edge_index[0], col=data.edge_index[1],
                value=torch.arange(E, device=data.edge_index.device),
                sparse_sizes=(N, N))
            adj, partptr, perm = adj.partition(num_parts, recursive)

            if save_dir is not None:
                torch.save((adj, partptr, perm), path)

            if log:  # pragma: no cover
                print('Done!', file=sys.stderr)

        self.data = self.__permute_data__(data, perm, adj)
        self.partptr = partptr
        self.perm = perm

    def __permute_data__(self, data, node_idx, adj):
        out = copy.copy(data)
        for key, value in data.items():
            if data.is_node_attr(key):
                cat_dim = data.__cat_dim__(key, value)
                out[key] = select(value, node_idx, dim=cat_dim)

        out.edge_index = None
        out.adj = adj

        return out

    def __len__(self):
        return self.partptr.numel() - 1

    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start

        data = copy.copy(self.data)
        adj, data.adj = data.adj, None

        adj = adj.narrow(0, start, length).narrow(1, start, length)
        edge_idx = adj.storage.value()

        for key, value in data:
            if key == 'num_nodes':
                data.num_nodes = length
            elif self.data.is_node_attr(key):
                cat_dim = self.data.__cat_dim__(key, value)
                data[key] = narrow(value, cat_dim, start, length)
            elif self.data.is_edge_attr(key):
                cat_dim = self.data.__cat_dim__(key, value)
                data[key] = select(value, edge_idx, dim=cat_dim)

        row, col, _ = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f'  num_parts={self.num_parts}\n'
                f')')


class ClusterLoader(torch.utils.data.DataLoader):
    r"""The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
    for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
    and their between-cluster links from a large-scale graph data object to
    form a mini-batch.

    .. note::

        Use :class:`~torch_geometric.loader.ClusterData` and
        :class:`~torch_geometric.loader.ClusterLoader` in conjunction to
        form mini-batches of clusters.
        For an example of using Cluster-GCN, see
        `examples/cluster_gcn_reddit.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py>`_ or
        `examples/cluster_gcn_ppi.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py>`_.

    Args:
        cluster_data (torch_geometric.loader.ClusterData): The already
            partioned data object.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, cluster_data, **kwargs):
        self.cluster_data = cluster_data

        super().__init__(range(len(cluster_data)), collate_fn=self.__collate__,
                         **kwargs)

    def __collate__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        start = self.cluster_data.partptr[batch].tolist()
        end = self.cluster_data.partptr[batch + 1].tolist()
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])

        data = copy.copy(self.cluster_data.data)

        adj, data.adj = self.cluster_data.data.adj, None
        adj = torch_sparse.cat(
            [adj.narrow(0, s, e - s) for s, e in zip(start, end)], dim=0)
        adj = adj.index_select(1, node_idx)
        row, col, edge_idx = adj.coo()

        for key, value in data:
            if key == 'num_nodes':
                data.num_nodes = node_idx.numel()
            elif self.cluster_data.data.is_node_attr(key):
                cat_dim = self.cluster_data.data.__cat_dim__(key, value)
                data[key] = select(value, node_idx, dim=cat_dim)
            elif self.cluster_data.data.is_edge_attr(key):
                cat_dim = self.cluster_data.data.__cat_dim__(key, value)
                data[key] = select(value, edge_idx, dim=cat_dim)

        data.edge_index = torch.stack([row, col], dim=0)

        return data
