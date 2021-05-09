from typing import Any, Optional, Iterable, Dict, List, Callable, Union
from torch_geometric.typing import NodeType, EdgeType

import copy
import warnings
from collections.abc import Sequence, Mapping, MutableMapping

from torch import Tensor


def recursive_apply_(data: Any, func: Callable) -> Any:
    if isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply_(d, func) for d in data]
    if isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(recursive_apply_(d) for d in data))
    if isinstance(data, Mapping):
        return {key: recursive_apply_(data[key], func) for key in data}
    if isinstance(data, Tensor):
        return func(data)
    try:
        return func(data)
    except:  # noqa
        return data


class MappingView(object):
    def __init__(self, mapping: Mapping, *args: List[str]):
        self._mapping = mapping
        self._args = set(args)

    def _keys(self) -> Iterable:
        if len(self._args) == 0:
            return self._mapping.keys()
        else:
            return set(self._mapping.keys()) & self._args

    def __len__(self) -> int:
        return len(self._keys())

    def __repr__(self) -> str:
        mapping = {key: self._mapping[key] for key in self._keys()}
        return f'{self.__class__.__name__}({mapping})'

    __class_getitem__ = classmethod(type([]))


class KeysView(MappingView):
    def __iter__(self) -> Iterable:
        yield from self._keys()


class ValuesView(MappingView):
    def __iter__(self) -> Iterable:
        for key in self._keys():
            yield self._mapping[key]


class ItemsView(MappingView):
    def __iter__(self):
        for key in self._keys():
            yield (key, self._mapping[key])


class BaseStorage(MutableMapping):
    # This class wraps a Python dictionary and extends it by the following:
    # 1. It allows attribute assignment:
    #    `storage.x = ...` rather than `storage['x'] = ...`
    # 2. It allows private non-dictionary attributes, e.g.:
    #    `storage.__dict__['key'] = ...
    # 3. It allows iterating over a subset of keys, e.g.:
    #    `storage.values('x', 'y')` or `storage.items('x', 'y')
    # 4. It adds additional functionality, e.g.:
    #    `storage.to_dict()` or `storage.apply_(...)`
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
        return self._data

    def clone(self, *args: List[str]):
        return copy.deepcopy(self)

    def apply_(self, func: Callable, *args: List[str]):
        for key, value in self.items(*args):
            self[key] = recursive_apply_(value, func)
        return self

    def contiguous(self, *args: List[str]):
        return self.apply(lambda x: x.contiguous(), *args)

    def to(self, device: Union[int, str], *args: List[str],
           non_blocking: bool = False):
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: List[str]):
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(self, device: Union[int, str], *args: List[str],
             non_blocking: bool = False):
        return self.apply(lambda x: x.cuda(non_blocking=non_blocking), *args)

    def pin_memory(self, *args: List[str]):
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: List[str]):
        return self.apply(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: List[str]):
        return self.apply(lambda x: x.detach_(), *args)

    def detach(self, *args: List[str]):
        return self.apply(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: List[str], requires_grad: bool = True):
        return self.apply(
            lambda x: x.requires_grad_(requires_grad=requires_grad), *args)


class NodeStorage(BaseStorage):
    def __init__(
        self,
        dict: Optional[Dict[str, Any]] = None,
        key: Optional[NodeType] = None,
        parent: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(dict, **kwargs)
        self.__dict__['_key'] = key
        self.__dict__['_parent'] = parent

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
            f"Please explicitly set 'num_nodes' as an attribute of "
            f"'data[{self._key}]'")
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
        dict: Optional[Dict[str, Any]] = None,
        key: Optional[EdgeType] = None,
        parent: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(dict, **kwargs)
        self.__dict__['_key'] = key
        self.__dict__['_parent'] = parent

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
            f"Please explicitly set 'num_edges' as an attribute of "
            f"'data[{self._key}]'")
        return None

    @property
    def num_edge_features(self):
        for value in self.values('edge_attr'):
            return 1 if value.dim() == 1 else value.size(-1)
        return 0

    @property
    def num_features(self) -> int:
        return self.num_edge_features

    def contains_isolated_nodes(self) -> bool:
        raise NotImplementedError

    def contains_self_loops(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def is_directed(self) -> bool:
        return not self.is_undirected


class GlobalStorage(NodeStorage, EdgeStorage):
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
    def num_features(self) -> int:
        return self.num_node_features
