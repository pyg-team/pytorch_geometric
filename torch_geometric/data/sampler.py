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
    def __init__(self, n_id):
        self.n_id = n_id
        self.__last_n_id__ = n_id
        self.blocks = []

    @property
    def batch_size(self):
        return self.n_id.size(0)

    def append(self, n_id, e_id, edge_index):
        size = (self.__last_n_id__.size(0), n_id.size(0))
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
        info = '<-'.join([str(n_id.size(0)) for n_id in n_ids])
        return '{}({})'.format(self.__class__.__name__, info)


class NeighborSampler(object):
    def __init__(self,
                 data,
                 size,
                 num_hops,
                 batch_size=1,
                 subset=None,
                 shuffle=False,
                 flow='source_to_target',
                 drop_last=False):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if subset is None:
            subset = torch.arange(data.num_nodes)
        if subset.dtype == torch.uint8:
            subset = subset.nonzero().view(-1)
        self.subset = subset

        flow = 'target_to_source'  # TODO
        # YOUR DATA NEEDS TO BE COALESCED!!!
        assert flow in ['source_to_target', 'target_to_source']
        i, j = (0, 1) if flow == 'target_to_source' else (1, 0)

        self.data = data
        self.Data = data.__class__
        deg = degree(data.edge_index[i], data.num_nodes, dtype=torch.long)
        self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])
        self.col = data.edge_index[j]
        self.tmp = torch.empty(data.num_nodes, dtype=torch.long)

        self.size = repeat(size, num_hops)
        self.num_hops = num_hops

    def __len__(self):
        if self.drop_last:
            return self.subset.size(0) // self.batch_size
        return (self.subset.size(0) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            self.subset = self.subset[torch.randperm(self.subset.size(0))]

        for i in range(len(self)):
            n_id = self.subset[i * self.batch_size:(i + 1) * self.batch_size]
            data_flow = DataFlow(n_id)
            self.tmp[n_id] = torch.arange(n_id.size(0))

            for l in range(self.num_hops):
                n_id, e_id = sampler(n_id, self.cumdeg, self.col, self.size[l])

                row, col = self.data.edge_index[:, e_id]
                row = self.tmp[row]
                self.tmp[n_id] = torch.arange(n_id.size(0))
                col = self.tmp[col]
                edge_index = torch.stack([row, col], dim=0)

                data_flow.append(n_id, e_id, edge_index)

            yield data_flow
