from __future__ import division

import torch
from torch_cluster import neighbor_sampler as sampler
from torch_geometric.utils import degree
from torch_geometric.utils.repeat import repeat

from .data import size_repr


class Block(object):
    def __init__(self, n_id, e_id, edge_index, size):
        self.n_id = n_id
        self.e_id = e_id
        self.edge_index = edge_index
        self.size = size

    def __repr__(self):
        info = [(key, getattr(self, key)) for key in self.__dict__]
        info = ['{}={}'.format(key, size_repr(item)) for key, item in info]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))


class DataFlow(object):
    def __init__(self, n_id, flow='source_to_target'):
        self.n_id = n_id
        self.flow = flow
        self.__last_n_id__ = n_id
        self.blocks = []

    @property
    def batch_size(self):
        return self.n_id.size(0)

    def append(self, n_id, e_id, edge_index):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        size = [None, None]
        size[i] = self.__last_n_id__.size(0)
        size[j] = n_id.size(0)
        block = Block(n_id, e_id, edge_index, size)
        self.blocks.append(block)
        self.__last_n_id__ = n_id

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[::-1][idx]

    def __iter__(self):
        for block in self.blocks[::-1]:
            yield block

    def to(self, device):
        for block in self.blocks:
            block.edge_index = block.edge_index.to(device)
        return self

    def __repr__(self):
        n_ids = [self.n_id] + [block.n_id for block in self.blocks]
        sep = '<-' if self.flow == 'source_to_target' else '->'
        info = sep.join([str(n_id.size(0)) for n_id in n_ids])
        return '{}({})'.format(self.__class__.__name__, info)


class NeighborSampler(object):
    def __init__(self,
                 data,
                 size,
                 num_hops,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 flow='source_to_target'):

        self.data = data
        self.size = repeat(size, num_hops)
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.flow = flow

        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)

        self.edge_index_i, self.e_assoc = data.edge_index[self.i].sort()
        self.edge_index_j = data.edge_index[self.j, self.e_assoc]
        deg = degree(self.edge_index_i, data.num_nodes, dtype=torch.long)
        self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])

        self.tmp = torch.empty(data.num_nodes, dtype=torch.long)

    def __len__(self):
        if self.drop_last:
            return self.subset.size(0) // self.batch_size
        return (self.subset.size(0) + self.batch_size - 1) // self.batch_size

    def __subsets__(self, subset=None):
        if subset is None and not self.shuffle:
            subset = torch.arange(self.data.num_nodes, dtype=torch.long)
        elif subset is None and self.shuffle:
            subset = torch.randperm(self.data.num_nodes)
        else:
            if subset.dtype == torch.uint8:
                subset = subset.nonzero().view(-1)
            if self.shuffle:
                subset = subset[torch.randperm(subset.size(0))]

        subsets = torch.split(subset, self.batch_size)
        if self.drop_last and subsets[-1].size(0) < self.batch_size:
            subsets = subsets[:-1]
        assert len(subsets) > 0
        return subsets

    def __call__(self, subset=None):
        for n_id in self.__subsets__(subset):
            data_flow = DataFlow(n_id, self.flow)
            self.tmp[n_id] = torch.arange(n_id.size(0), dtype=torch.long)

            for l in range(self.num_hops):
                n_id, e_id = sampler(n_id, self.cumdeg, self.edge_index_j,
                                     self.size[l])
                e_id = self.e_assoc[e_id]

                edges = [None, None]
                edges[self.i] = self.tmp[self.data.edge_index[self.i, e_id]]
                self.tmp[n_id] = torch.arange(n_id.size(0))
                edges[self.j] = self.tmp[self.data.edge_index[self.j, e_id]]
                edge_index = torch.stack(edges, dim=0)

                data_flow.append(n_id, e_id, edge_index)

            yield data_flow
