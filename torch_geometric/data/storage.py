from typing import Any, Optional, Iterable, Dict, List

import copy
import warnings
import collections

from typing import Union, Tuple, Callable, Optional

NodeType = str
EdgeType = Tuple[str, str, str]
QueryType = Union[str, NodeType, EdgeType, Tuple[str, str]]
AttrType = Union[str, Tuple[str, Callable]]


class BaseStorage(collections.abc.MutableMapping):
    # This class extends the Python dictionary class by the following:
    # 1. It allows attribute assignment:
    #    `storage.x = ...` rather than `storage['x'] = ...`
    # 2. We can iterate over a subset of keys, e.g.:
    #    `storage.items('x', 'y')` or `storage.values('x', 'y')
    # 3. It allows for private non-dict attributes, e.g.:
    #    `storage.__dict__['key'] = ...
    def __init__(
        self,
        dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.__dict__['_data'] = {}
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

    def __call__(self, *args: List[str]) -> Iterable:
        keys = self.keys()
        for key in keys if len(args) == 0 else (set(keys) & set(args)):
            yield key, self[key]

    def __repr__(self) -> str:
        return repr(self._data)

    def __copy__(self):
        return self.__class__(copy.copy(self._data), self._key, self._parent)

    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self._data, memo), self._key,
                              self._parent)


class Storage(BaseStorage):
    def __init__(
        self,
        dict: Optional[Dict[str, Any]] = None,
        key: Optional[Union[NodeType, EdgeType]] = None,
        parent: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(dict, **kwargs)
        self.__dict__['_key'] = key
        self.__dict__['_parent'] = parent

    @property
    def num_nodes(self) -> Optional[int]:
        for _, item in self('num_nodes'):
            return item
        for _, item in self('x', 'pos'):
            return item.size(-2)
        for _, item in self('adj'):
            return item.size(0)
        for _, item in self('adj_t'):
            return item.size(1)
        warnings.warn((
            f"Unable to infer 'num_nodes' from attributes {set(self.keys())}. "
            f"Please explicitly set 'num_nodes' as an attribute"))
        for _, item in self('edge_index'):
            return int(max(item)) + 1
        for _, item in self('face'):
            return int(max(item)) + 1
        return None

    @property
    def num_edges(self) -> Optional[int]:
        for _, item in self('num_edges'):
            return item
        for _, item in self('edge_index'):
            return item.size(-1)
        for _, item in self('adj', 'adj_t'):
            return item.nnz()
        warnings.warn((
            f"Unable to infer 'num_edges' from attributes {set(self.keys())}. "
            f"Please explicitly set 'num_edges' as an attribute"))
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
