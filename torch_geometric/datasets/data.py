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

    @property
    def num_nodes(self):
        if self.x is not None:
            return self.x.size(0)
        elif self.pos is not None:
            return self.pos.size(0)
        else:
            return None

    @property
    def num_edges(self):
        if self.edge_index is not None:
            return self.edge_index.size(1)
        elif self.edge_attr is not None:
            return self.edge_attr.size(0)
        else:
            return None

    def cuda(self, *props):
        def func(x):
            return x.cuda() if torch.cuda.is_available() else x

        return self._transfer(func, props)

    def cpu(self, *props):
        return self._transfer(lambda x: x.cpu(), props)

    def to_variable(self, *props):
        props = ('x', 'edge_attr', 'y') if not props else props
        return self._transfer(lambda x: Variable(x), props)

    def to_tensor(self, *props):
        def func(x):
            return x if torch.is_tensor(x) else x.data

        return self._transfer(func, props)

    def _transfer(self, func, props):
        props = self.__dict__.keys() if not props else props
        for prop in props:
            if getattr(self, prop) is not None:
                setattr(self, prop, func(getattr(self, prop)))
        return self

    def __repr__(self):
        info = self.__dict__
        info = {k: v for k, v in info.items() if v is not None}
        info = {k: list(v.size()) for k, v in info.items()}
        info = ['{}={}'.format(k, v) for k, v in info.items()]
        return 'Data({})'.format(', '.join(info))
