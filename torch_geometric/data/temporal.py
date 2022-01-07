from typing import NamedTuple, Dict, Any, List

import copy

import torch
import numpy as np

from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import EdgeStorage, NodeStorage, BaseStorage, GlobalStorage


class TemporalData(BaseData):
    def __init__(self, src=None, dst=None, t=None, msg=None, y=None, **kwargs):
        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg
        self.y = y

        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def __prepare_non_str_idx(idx):
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        if isinstance(idx, (list, tuple)):
            idx = torch.tensor(idx)
        elif isinstance(idx, slice):
            pass
        elif isinstance(idx, torch.Tensor) and (idx.dtype == torch.long
                                                or idx.dtype == torch.bool):
            pass
        else:
            raise IndexError(
                f'Only strings, integers, slices (`:`), list, tuples, and '
                f'long or bool tensors are valid indices (got '
                f'{type(idx).__name__}).')
        return idx

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._store[idx]

        prepared_idx = self.__prepare_non_str_idx(idx)

        data = copy.copy(self._store)
        for key, item in data:
            if item.shape[0] == self.num_events:
                data[key] = item[prepared_idx]
        return data

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        self._store[key] = value

    def __delitem__(self, idx):
        if isinstance(idx, str) and idx in self._store:
            del self._store[idx]

        prepared_idx = self.__prepare_non_str_idx(idx)

        for key, item in self._store:
            if item.shape[0] == self.num_events:
                del item[prepared_idx]

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def stores_as(self, data: 'TemporalData'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    def debug(self):
        pass # TODO

    @property
    def keys(self):
        return [key for key in self._store.keys() if self[key] is not None]

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

    def train_val_test_split(self, val_ratio=0.15, test_ratio=0.15):
        val_time, test_time = np.quantile(
            self.t.cpu().numpy(),
            [1. - val_ratio - test_ratio, 1. - test_ratio])

        val_idx = int((self.t <= val_time).sum())
        test_idx = int((self.t <= test_time).sum())

        return self[:val_idx], self[val_idx:test_idx], self[test_idx:]

    def seq_batches(self, batch_size):
        for start in range(0, self.num_events, batch_size):
            yield self[start:start + batch_size]

    def __cat_dim__(self, key, value, *args, **kwargs):
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        return 0

    def __repr__(self):
        cls = str(self.__class__.__name__)
        shapes = ', '.join([f'{k}={list(v.shape)}' for k, v in self])
        return f'{cls}({shapes})'
