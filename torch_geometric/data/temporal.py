import torch
import numpy as np


class TemporalData(object):
    def __init__(self, src=None, dst=None, t=None, x=None, y=None, **kwargs):
        self.src = src
        self.dst = dst
        self.t = t
        self.x = x
        self.y = y
        for key, item in kwargs.items():
            self[key] = item

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

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
            if key in self:
                yield key, self[key]

    @property
    def num_nodes(self):
        return max(int(self.src.max()), int(self.dst.max())) + 1

    @property
    def num_events(self):
        return self.src.size(0)

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def to(self, device, *keys, **kwargs):
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def __cat_dim__(self, key, value):
        return 0

    def __inc__(self, key, value):
        return 0

    def train_val_test_split(self, val_ratio=0.15, test_ratio=0.15):
        val_time, test_time = np.quantile(self.t.cpu().numpy(), [0.7, 0.85])

    def __repr__(self):
        cls = str(self.__class__.__name__)
        shapes = ', '.join([f'{k}={list(v.shape)}' for k, v in self])
        return f'{cls}({shapes})'
