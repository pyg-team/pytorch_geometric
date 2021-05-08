from typing import Any, Optional, Iterable, Dict
from collections import UserDict
from collections.abc import MutableMapping
import collections

import copy
import warnings

from typing import Union, Tuple, Callable, Optional

NodeType = str
EdgeType = Tuple[str, str, str]
QueryType = Union[str, NodeType, EdgeType, Tuple[str, str]]
AttrType = Union[str, Tuple[str, Callable]]


class Storage2(collections.abc.MutableMapping):
    # This class extends the Python dictionary class by the following:
    # 1. It allows attribute assignment:
    #    `storage.x = ...` rather than `storage['x'] = ...`
    # 2. We can iterate over a subset of keys:
    #    `storage.items('x', 'y')` or `storage.values('x', 'y')
    # 3. It hides private attributes to the user, e.g.:
    #    `storage._key` and `storage._parent`
    def __init__(
        self,
        dict: Optional[Dict[str, Any]] = None,
        key: Optional[Union[NodeType, EdgeType]] = None,
        parent: Optional[Any] = None,
        **kwargs,
    ):
        self.__dict__['_data'] = {}
        self.__dict__['_key'] = key
        self.__dict__['_parent'] = parent

        if dict is not None:
            self.update(dict)
        if kwargs:
            self.update(kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key: str) -> Optional[Any]:
        return self._data.get(key, None)

    def __getattr__(self, key: str) -> Optional[Any]:
        return self[key]

    def __setitem__(self, key: str, item: Optional[Any]):
        if item is None:
            del self[key]
        self._data[key] = item

    def __setattr__(self, key: str, item: Optional[Any]):
        if key in self.__dict__:
            self.__dict__[key] = item
        self[key] = item

    def __delitem__(self, key: str):
        if key in self._data:
            del self._data[key]

    def __delattr__(self, key: str):
        if key in self.__dict__:
            del self.__dict__[key]
        del self[key]

    def __iter__(self) -> Iterable:
        return iter(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __copy__(self):
        return self.__class__(copy.copy(self._data), self._key, self._parent)

    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self._data, memo), self._key,
                              self._parent)


class Storage(dict):
    # This class extends the Python dictionary class by the following:
    # 1. It allows attribute assignment:
    #    `storage.x = ...` rather than `storage['x'] = ...`
    # 2. We can only iterate over specific keys:
    #    `storage.items('x', 'y')` or `storage.values('x', 'y')
    def __getattr__(self, key: str) -> Any:
        return self.get(key, None)

    def __setattr__(self, key: str, item: Any):
        if item is None and key in self:
            del self[key]
        elif item is not None:
            self[key] = item

    def __delattr__(self, key: str):
        if key in self:
            del self[key]

    def __copy__(self):
        out = self.__class__()
        out.update(self)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, item in self.items():
            out[key] = copy.deepcopy(item, memo)
        return out

    def keys(self, *args) -> Iterable:
        if len(args) > 0:
            return {key: self[key] for key in args if key in self}.keys()
        else:
            return super().keys()

    def values(self, *args) -> Iterable:
        if len(args) > 0:
            return {key: self[key] for key in args if key in self}.values()
        else:
            return super().values()

    def items(self, *args) -> Iterable:
        if len(args) > 0:
            return {key: self[key] for key in args if key in self}.items()
        else:
            return super().items()

    @property
    def num_nodes(self) -> Optional[int]:
        for value in self.values('num_nodes'):
            return value
        for value in self.values('x', 'pos'):
            return value.size(-2)
        for value in self.values('adj'):
            return value.size(0)
        for value in self.values('adj_t'):
            return value.size(1)
        warnings.warn((
            f"Unable to infer 'num_nodes' from attributes {set(self.keys())}. "
            f"Please consider explicitly setting 'num_nodes' as attribute"))
        for value in self.values('edge_index'):
            return int(max(value)) + 1
        for value in self.values('face'):
            return int(max(value)) + 1
        return None

    @property
    def num_edges(self) -> Optional[int]:
        for value in self.values('num_edges'):
            return value
        for value in self.values('edge_index'):
            return value.size(-1)
        for value in self.values('adj', 'adj_t'):
            return value.nnz()
        warnings.warn((
            f"Unable to infer 'num_edges' from attributes {set(self.keys())}. "
            f"Please consider explicitly setting 'num_edges' as attribute"))
        return None


class NodeStorage(Storage):
    @property
    def num_edges(self) -> Optional[int]:
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute 'num_edges'")


class EdgeStorage(Storage):
    @property
    def num_nodes(self) -> Optional[int]:
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute 'num_nodes'")
