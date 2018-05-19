import torch


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

    def keys(self):
        return [key for key in self.__dict__.keys() if self[key] is not None]

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        for key in self.keys():
            yield key, self[key]

    def __call__(self, *keys):
        for key in self.keys() if not keys else keys:
            if self[key] is not None:
                yield key, self[key]

    def cat_dim(self, key):
        return -1 if self[key].dtype == torch.long else 0

    def should_cumsum(self, key):
        return self[key].dim() > 1 and self[key].dtype == torch.long

    @property
    def num_nodes(self):
        for key, item in self('x', 'pos'):
            return item.size(self.cat_dim(key))
        if self.edge_index is not None:
            return self.edge_index.max().item() + 1
        return None

    @property
    def num_edges(self):
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.cat_dim(key))
        return None

    @property
    def num_graphs(self):
        if self.batch is not None:
            return self.batch.max().item() + 1
        # NOTE: return `1` might not be true for datasets. However, this should
        # not be problematic, because the user only operates on singular or
        # batched data.
        return 1

    @property
    def contains_isolated_nodes(self):
        raise NotImplementedError

    @property
    def contains_self_loops(self):
        raise NotImplementedError

    @property
    def is_directed(self):
        raise NotImplementedError

    def apply(self, func, *keys):
        for key, item in self(*keys):
            self[key] = func(item)
        return self

    def contiguous(self, *keys):
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        return self.apply(lambda x: x.to(device), *keys)

    def __repr__(self):
        keys = sorted(self.keys())
        info = ['{}={}'.format(key, list(self[key].size())) for key in keys]
        return 'Data({})'.format(', '.join(info))
