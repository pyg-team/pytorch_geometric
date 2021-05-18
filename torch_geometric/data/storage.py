from typing import (Any, Optional, Iterable, Dict, List, Callable, Union,
                    Tuple, NamedTuple)
from torch_geometric.typing import NodeType, EdgeType

import copy
import warnings
from inspect import signature
from collections import namedtuple
from collections.abc import Sequence, Mapping, MutableMapping

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from .view import KeysView, ValuesView, ItemsView


def recursive_apply(data: Any, func: Callable) -> Any:
    if isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, func) for d in data]
    if isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(recursive_apply(d) for d in data))
    if isinstance(data, Mapping):
        return {key: recursive_apply(data[key], func) for key in data}
    if isinstance(data, Tensor):
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
    #    `storage.to_dict()` or `storage.apply(...)`
    def __init__(self, mapping: Optional[Dict[str, Any]] = None, **kwargs):
        self.__dict__['_mapping'] = {}
        if mapping is not None:
            self.update(mapping)
        if kwargs:
            self.update(kwargs)

    def __len__(self):
        return len(self._mapping)

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setitem__(self, key: str, value: Any):
        if value is None:
            del self[key]
        else:
            self._mapping[key] = value

    def __setattr__(self, key: str, value: Any):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delitem__(self, key: str):
        if key in self._mapping:
            del self._mapping[key]

    def __delattr__(self, key: str):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            del self[key]

    def __iter__(self) -> Iterable:
        return iter(self._mapping)

    def __copy__(self):
        out = self.__class__()
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

    def __repr__(self) -> str:
        return repr(self._mapping)

    # Allow iterating over subsets ############################################

    def keys(self, *args: List[str]) -> KeysView:
        return KeysView(self._mapping, *args)

    def values(self, *args: List[str]) -> ValuesView:
        return ValuesView(self._mapping, *args)

    def items(self, *args: List[str]) -> ItemsView:
        return ItemsView(self._mapping, *args)

    # Additional functionality ################################################

    def to_dict(self) -> Dict[str, Any]:
        return copy.copy(self._mapping)

    def to_namedtuple(self) -> NamedTuple:
        field_names = list(self.keys())
        StorageTuple = namedtuple('StorageTuple', field_names)
        return StorageTuple(*[self[key] for key in field_names])

    def clone(self, *args: List[str]):
        return copy.deepcopy(self)

    def apply(self, func: Callable, *args: List[str]):
        for key, value in self.items(*args):
            self[key] = recursive_apply(value, func)
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
    def __init__(self, _parent: Any, _key: Optional[NodeType] = None,
                 mapping: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(mapping, **kwargs)
        assert _key is None or isinstance(_key, NodeType)
        self.__dict__['_key'] = _key
        self.__dict__['_parent'] = _parent

    def __copy__(self):
        out = self.__class__(self._parent)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__(self._parent)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    @property
    def num_nodes(self) -> Optional[int]:
        for value in self.values('num_nodes'):
            return value
        for key, value in self.items('x', 'pos', 'batch'):
            num_args = len(signature(self._parent.__cat_dim__).parameters)
            args = (key, value, self)[:num_args]
            return value.size(self._parent.__cat_dim__(*args))
        for value in self.values('adj'):
            return value.size(0)
        for value in self.values('adj_t'):
            return value.size(1)
        warnings.warn(
            f"Unable to accurately infer 'num_nodes' from the attribute set "
            f"{set(self.keys())}. Please explicitly set 'num_nodes' as an "
            f"attribute of " +
            ("'data'" if self._key is None else f"'data[{self._key}]'") +
            " to suppress this warning")
        for value in self.values('edge_index'):
            return int(value.max()) + 1
        for value in self.values('face'):
            return int(value.max()) + 1
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
    def __init__(self, _parent: Any, _key: Optional[EdgeType] = None,
                 mapping: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(mapping, **kwargs)
        assert _key is None or (isinstance(_key, tuple) and len(_key) == 3)
        self.__dict__['_key'] = _key
        self.__dict__['_parent'] = _parent

    @property
    def _edge_index(self) -> Optional[Tensor]:
        for value in self.values('edge_index'):
            return value
        for value in self.values('adj'):
            if isinstance(value, SparseTensor):
                return torch.stack(value.coo()[:2], dim=0)
        for value in self.values('adj_t'):
            if isinstance(value, SparseTensor):
                return torch.stack(value.coo()[:2][::-1], dim=0)
        return None

    def __copy__(self):
        out = self.__class__(self._parent)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__(self._parent)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    @property
    def num_edges(self) -> Optional[int]:
        for value in self.values('num_edges'):
            return value
        for key, value in self.items('edge_index', 'edge_attr'):
            num_args = len(signature(self._parent.__cat_dim__).parameters)
            args = (key, value, self)[:num_args]
            return value.size(self._parent.__cat_dim__(*args))
        for value in self.values('adj', 'adj_t'):
            return value.nnz()
        warnings.warn(
            f"Unable to accurately infer 'num_edges' from the attribute set "
            f"{set(self.keys())}. Please explicitly set 'num_edges' as an "
            f"attribute of " +
            ("'data'" if self._key is None else f"'data[{self._key}]'") +
            " to suppress this warning")
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
            return value.size(0), value.size(1)
        for value in self.values('adj_t'):
            return value.size(1), value.size(0)

        if self._key is None:
            raise NameError(
                "Unable to infer 'size' without explicit '_key' assignment")

        return (self._parent[self._key[0]].num_nodes,
                self._parent[self._key[-1]].num_nodes)

    def is_coalesced(self) -> bool:
        raise NotImplementedError

    def coalesce(self):
        raise NotImplementedError

    def has_isolated_nodes(self) -> bool:
        edge_index = self._edge_index
        if edge_index is not None:
            num_nodes = self.size()[-1]
            if num_nodes is None:
                raise NameError("Unable to infer 'num_nodes'")
            return torch.unique(edge_index[1]).numel() < num_nodes
        return True

    def has_self_loops(self) -> bool:
        if self._key is None or self._key[0] == self._key[-1]:
            edge_index = self._edge_index
            if edge_index is not None:
                return int((edge_index[0] == edge_index[1]).sum()) > 0
        return False

    def is_undirected(self) -> bool:
        if self._key is None or self._key[0] == self._key[-1]:
            from torch_geometric.utils import is_undirected
            return is_undirected(self.edge_index)
            raise NotImplementedError
        return False

    def is_directed(self) -> bool:
        return not self.is_undirected()


class GlobalStorage(NodeStorage, EdgeStorage):
    def __init__(self, _parent: Any, mapping: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(_parent=_parent, mapping=mapping, **kwargs)

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def size(self) -> Tuple[Optional[int], Optional[int]]:
        return [self.num_nodes, self.num_nodes]
