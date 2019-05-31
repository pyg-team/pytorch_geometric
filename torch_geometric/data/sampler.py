from __future__ import division

import torch
from torch_cluster import neighbor_sampler
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
        block = Block(n_id, e_id, edge_index, tuple(size))
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
                 add_self_loop=False,
                 flow='source_to_target'):

        self.data = data
        self.size = repeat(size, num_hops)
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.add_self_loop = add_self_loop
        self.flow = flow

        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)

        self.edge_index_i, self.e_assoc = data.edge_index[self.i].sort()
        self.edge_index_j = data.edge_index[self.j, self.e_assoc]
        deg = degree(self.edge_index_i, data.num_nodes, dtype=torch.long)
        self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])

        self.tmp = torch.empty(data.num_nodes, dtype=torch.long)

    def __get_batches__(self, subset=None):
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
        for n_id in self.__get_batches__(subset):
            data_flow = DataFlow(n_id, self.flow)
            self.tmp[n_id] = torch.arange(n_id.size(0), dtype=torch.long)

            for l in range(self.num_hops):
                e_id = neighbor_sampler(n_id, self.cumdeg, self.size[l])

                new_n_id = self.edge_index_j.index_select(0, e_id)
                if self.add_self_loop:
                    new_n_id = torch.cat([new_n_id, n_id], dim=0)
                new_n_id = new_n_id.unique(sorted=False)
                e_id = self.e_assoc[e_id]

                edges = [None, None]

                edge_index_i = self.data.edge_index[self.i, e_id]
                if self.add_self_loop:
                    edge_index_i = torch.cat([edge_index_i, n_id], dim=0)
                edges[self.i] = self.tmp[edge_index_i]

                self.tmp[new_n_id] = torch.arange(new_n_id.size(0))
                edge_index_j = self.data.edge_index[self.j, e_id]
                if self.add_self_loop:
                    edge_index_j = torch.cat([edge_index_j, n_id], dim=0)
                edges[self.j] = self.tmp[edge_index_j]

                edge_index = torch.stack(edges, dim=0)

                # Remove the edge identifier when adding self-loops to prevent
                # misused behavior.
                e_id = None if self.add_self_loop else e_id
                n_id = new_n_id

                data_flow.append(n_id, e_id, edge_index)

            yield data_flow
