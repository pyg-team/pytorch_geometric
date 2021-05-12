from typing import (Any, Optional, Iterable, Dict, List, Callable, Union,
                    Tuple, NamedTuple)
from torch_geometric.typing import NodeType, EdgeType

import copy
import warnings
from collections import namedtuple
from collections.abc import Sequence, Mapping, MutableMapping

import torch

from .view import KeysView, ValuesView, ItemsView


def recursive_apply_(data: Any, func: Callable) -> Any:
    if isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply_(d, func) for d in data]
    if isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(recursive_apply_(d) for d in data))
    if isinstance(data, Mapping):
        return {key: recursive_apply_(data[key], func) for key in data}
    if isinstance(data, torch.Tensor):
        return func(data)
    try:
        return func(data)
    except:  # noqa
        return data


class BaseStorage(MutableMapping):
    # This class wraps a Python dictionary and extends it by the following:
    # 1. It allows attribute assignment:
    #    `storage.x = ...` rather than `storage['x'] = ...`
    # 2. It allows private non-dictionary attributes, e.g.:
    #    `storage.__dict__[{key}] = ...` accesible via `storage.{key}`
    # 3. It allows iterating over a subset of keys, e.g.:
    #    `storage.values('x', 'y')` or `storage.items('x', 'y')
    # 4. It adds additional functionality, e.g.:
    #    `storage.to_dict()` or `storage.apply_(...)`
    def __init__(
        self,
        dictionary: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.__dict__['_data'] = {}
        if dictionary is not None:
            self.update(dictionary)
        if kwargs:
            self.update(kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Optional[Any]:
        return self[key]

    def __setitem__(self, key: str, value: Optional[Any]):
        if value is None:
            del self[key]
        else:
            self._data[key] = value

    def __setattr__(self, key: str, value: Optional[Any]):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delitem__(self, key: str):
        if key in self._data:
            del self._data[key]

    def __delattr__(self, key: str):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            del self[key]

    def __iter__(self) -> Iterable:
        return iter(self._data)

    def __copy__(self):
        out = self.__class__()
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._data = copy.copy(out._data)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._data = copy.deepcopy(out._data, memo)
        return out

    def __repr__(self) -> str:
        return repr(self._data)

    # Allow iterating over subsets ############################################

    def keys(self, *args: List[str]) -> KeysView:
        return KeysView(self._data, *args)

    def values(self, *args: List[str]) -> ValuesView:
        return ValuesView(self._data, *args)

    def items(self, *args: List[str]) -> ItemsView:
        return ItemsView(self._data, *args)

    # Additional functionality ################################################

    def to_dict(self) -> Dict[str, Any]:
        return copy.copy(self._data)

    def to_namedtuple(self) -> NamedTuple:
        typename = 'NamedTuple'
        if self._key is not None:
            typename = '__'.join(self._key)
        field_names = list(self.keys())
        DataTuple = namedtuple(typename, field_names)
        return DataTuple(*[self[key] for key in field_names])

    def clone(self, *args: List[str]):
        return copy.deepcopy(self)

    def apply_(self, func: Callable, *args: List[str]):
        for key, value in self.items(*args):
            self[key] = recursive_apply_(value, func)
        return self

    def contiguous(self, *args: List[str]):
        return self.apply_(lambda x: x.contiguous(), *args)

    def to(self, device: Union[int, str], *args: List[str],
           non_blocking: bool = False):
        return self.apply_(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: List[str]):
        return self.apply_(lambda x: x.cpu(), *args)

    def cuda(self, device: Union[int, str], *args: List[str],
             non_blocking: bool = False):
        return self.apply_(lambda x: x.cuda(non_blocking=non_blocking), *args)

    def pin_memory(self, *args: List[str]):
        return self.apply_(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: List[str]):
        return self.apply_(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: List[str]):
        return self.apply_(lambda x: x.detach_(), *args)

    def detach(self, *args: List[str]):
        return self.apply_(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: List[str], requires_grad: bool = True):
        return self.apply_(
            lambda x: x.requires_grad_(requires_grad=requires_grad), *args)


class NodeStorage(BaseStorage):
    def __init__(
        self,
        _parent: Any,
        _key: Optional[NodeType] = None,
        dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(dict, **kwargs)
        assert _key is None or isinstance(_key, NodeType)
        self.__dict__['_key'] = _key
        self.__dict__['_parent'] = _parent

    @property
    def num_nodes(self) -> Optional[int]:
        for value in self.values('num_nodes'):
            return value
        for key, value in self.items('x', 'pos', 'batch'):
            return value.size(self._parent.__cat_dim__(key, value))
        for value in self.values('adj'):
            return value.size(0)
        for value in self.values('adj_t'):
            return value.size(1)
        warnings.warn(
            f"Unable to infer 'num_nodes' from attributes {set(self.keys())}. "
            f"Please explicitly set 'num_nodes' as an attribute" +
            f" of 'data[{self._key}]'" if self._key is not None else "")
        for value in self.values('edge_index'):
            return int(max(value)) + 1
        for value in self.values('face'):
            return int(max(value)) + 1
        return None

    @property
    def num_node_features(self) -> int:
        for value in self.values('x'):
            return 1 if value.dim() == 1 else value.size(-1)
        return 0

    @property
    def num_features(self) -> int:
        return self.num_node_features


class EdgeStorage(BaseStorage):
    def __init__(
        self,
        _parent: Any,
        _key: Optional[EdgeType] = None,
        dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(dict, **kwargs)
        assert _key is None or (isinstance(_key, tuple) and len(_key) == 3)
        self.__dict__['_key'] = _key
        self.__dict__['_parent'] = _parent

    @property
    def num_edges(self) -> Optional[int]:
        for value in self.values('num_edges'):
            return value
        for key, value in self.items('edge_index', 'edge_attr'):
            return value.size(self._parent.__cat_dim__(key, value))
        for value in self.values('adj', 'adj_t'):
            return value.nnz()
        warnings.warn(
            f"Unable to infer 'num_edges' from attributes {set(self.keys())}. "
            f"Please explicitly set 'num_edges' as an attribute" +
            f" of 'data[{self._key}]'" if self._key is not None else "")
        return None

    @property
    def num_edge_features(self):
        for value in self.values('edge_attr'):
            return 1 if value.dim() == 1 else value.size(-1)
        return 0

    @property
    def num_features(self) -> int:
        return self.num_edge_features

    def size(self) -> Tuple[Optional[int], Optional[int]]:
        for value in self.values('adj'):
            return [value.size(0), value.size(1)]
        for value in self.values('adj_t'):
            return [value.size(1), value.size(0)]

        if self._key is None:
            raise NameError(
                "Unable to infer 'size' without explicit '_key' assignment")

        return [
            self._parent[self._key[0]].num_nodes,
            self._parent[self._key[-1]].num_nodes
        ]

    def has_isolated_nodes(self) -> bool:
        raise NotImplementedError

    def has_self_loops(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def is_directed(self) -> bool:
        return not self.is_undirected


class GlobalStorage(NodeStorage, EdgeStorage):
    def __init__(
        self,
        _parent: Any,
        dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(_parent, dict=dict, **kwargs)

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def size(self) -> Tuple[Optional[int], Optional[int]]:
        return [self.num_nodes, self.num_nodes]

    def has_isolated_nodes(self) -> bool:
        raise NotImplementedError

    def has_self_loops(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def is_directed(self) -> bool:
        return not self.is_undirected
