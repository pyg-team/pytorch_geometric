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

    @property
    def keys(self):
        keys = self.__dict__.keys()
        return [key for key in keys if getattr(self, key) is not None]

    def __call__(self, *keys):
        keys = self.keys if not keys else keys
        for key in self.keys:
            yield key, getattr(self, key)

    def __iter__(self):
        yield self()

    @property
    def num_nodes(self):
        for _, value in self('x', 'pos'):
            return value.size(0)
        raise ValueError('Can not compute number of nodes')

    @property
    def num_edges(self):
        for _, value in self('edge_attr'):
            return value.size(0)
        if self.edge_index is not None:
            return self.edge_index.size(1)
        raise ValueError('Can not compute number of edges')

    def _apply(self, func, *keys):
        for key, value in self(*keys):
            setattr(self, key, func(value))
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
        info = ['{}={}'.format(key, list(value.size())) for key, value in self]
        return 'Data({})'.format(', '.join(info))
