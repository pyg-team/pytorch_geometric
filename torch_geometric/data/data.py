import torch
from torch_geometric.utils import (contains_isolated_nodes,
                                   contains_self_loops, is_undirected)

from ..utils.num_nodes import maybe_num_nodes


class Data(object):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 pos=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos

    @staticmethod
    def from_dict(dictionary):
        data = Data()
        for key, item in dictionary.items():
            data[key] = item
        return data

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)

    @property
    def keys(self):
        return [key for key in self.__dict__.keys() if self[key] is not None]

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self.keys

    def __iter__(self):
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        for key in sorted(self.keys) if not keys else keys:
            if self[key] is not None:
                yield key, self[key]

    def cat_dim(self, key):
        return -1 if self[key].dtype == torch.long else 0

    @property
    def num_nodes(self):
        for key, item in self('x', 'pos'):
            return item.size(self.cat_dim(key))
        if self.edge_index is not None:
            return maybe_num_nodes(self.edge_index)
        return None

    @property
    def num_edges(self):
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.cat_dim(key))
        return None

    @property
    def num_features(self):
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_classes(self):
        return self.y.max().item() + 1 if self.y.dim() == 1 else self.y.size(1)

    def is_coalesced(self):
        row, col = self.edge_index
        index = self.num_nodes * row + col
        return self.row.size(0) == torch.unique(index).size(0)

    def contains_isolated_nodes(self):
        return contains_isolated_nodes(self.edge_index, self.num_nodes)

    def contains_self_loops(self):
        return contains_self_loops(self.edge_index)

    def is_undirected(self):
        return is_undirected(self.edge_index, self.num_nodes)

    def is_directed(self):
        return not self.is_undirected

    def apply(self, func, *keys):
        for key, item in self(*keys):
            self[key] = func(item)
        return self

    def contiguous(self, *keys):
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        return self.apply(lambda x: x.to(device), *keys)

    def __repr__(self):
        info = ['{}={}'.format(key, list(item.size())) for key, item in self]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))
