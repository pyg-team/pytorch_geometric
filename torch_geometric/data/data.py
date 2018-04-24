from torch import is_tensor
from torch.cuda import is_available as has_cuda
from torch.autograd import Variable


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
        for key in self.keys:
            yield key, self[key]

    def __call__(self, *keys):
        for key in self.keys if not keys else keys:
            yield key, self[key]

    @property
    def num_nodes(self):
        for _, item in self('x', 'pos'):
            return item.size(0)
        raise ValueError('Can\'t determine the number of nodes')

    def _apply(self, func, *keys):
        for key, item in self(*keys):
            setattr(self, key, func(item))
        return self

    def cuda(self, *keys):
        return self._apply(lambda x: x.cuda() if has_cuda() else x, *keys)

    def cpu(self, *keys):
        return self._apply(lambda x: x.cpu(), *keys)

    def to_variable(self, *keys):
        keys = ['x', 'edge_attr', 'y'] if not keys else keys
        return self._apply(lambda x: Variable(x), *keys)

    def to_tensor(self, *keys):
        return self._apply(lambda x: x if is_tensor(x) else x.data, *keys)

    def __repr__(self):
        info = ['{}={}'.format(key, list(item.size())) for key, item in self]
        return 'Data({})'.format(', '.join(info))
