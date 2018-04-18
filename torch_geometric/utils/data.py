import torch
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

    def __iter__(self):
        for key in self.__dict__.keys():
            if getattr(self, key) is not None:
                yield key, getattr(self, key)

    def __call__(self, *props):
        props = self.__dict__.keys() if not props else props
        for key in props:
            if getattr(self, key) is not None:
                yield key, getattr(self, key)

    def _apply(self, func, *props):
        for key, value in self(*props):
            setattr(self, key, func(value))
        return self

    @property
    def num_nodes(self):
        for _, value in self('x', 'pos'):
            return value.size(0)
        return None

    @property
    def num_edges(self):
        for _, value in self('edge_attr'):
            return value.size(0)
        return None if self.edge_index is None else self.edge_index.size(1)

    def cuda(self, *props):
        def func(x):
            return x.cuda() if torch.cuda.is_available() else x

        return self._apply(func, *props)

    def cpu(self, *props):
        return self._apply(lambda x: x.cpu(), *props)

    def to_variable(self, *props):
        props = ('x', 'edge_attr', 'y') if not props else props
        return self._apply(lambda x: Variable(x), *props)

    def to_tensor(self, *props):
        def func(x):
            return x if torch.is_tensor(x) else x.data

        return self._apply(func, *props)

    def __repr__(self):
        info = ['{}={}'.format(k, list(v.size())) for k, v in self]
        return 'Data({})'.format(', '.join(info))
