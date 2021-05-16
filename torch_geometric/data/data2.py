from typing import (Optional, Dict, Any, Tuple, Union, List, Iterable,
                    NamedTuple, Callable)
from torch_geometric.typing import NodeType, EdgeType, QueryType
from torch_geometric.deprecation import deprecated

import copy
from itertools import chain
from collections import namedtuple
from collections.abc import Sequence, Mapping

import torch
import numpy as np
from torch_sparse import SparseTensor

from .storage import NodeStorage, EdgeStorage, GlobalStorage

# Backward-compatibility issues:
# * `Data(x, edge_index)` breaks
# * `data.keys()` in favor of `data.keys`


def homogeneous_only(func: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        if self.is_heterogeneous:
            raise AttributeError(f"'{func.__name__}' is only supported in a "
                                 f"homogeneous data setting")
        return func(self, *args, **kwargs)

    return wrapper


class Data(object):
    def __init__(self, mapping: Optional[Dict[str, Any]] = None, **kwargs):
        self.__dict__['_global_store'] = GlobalStorage(_parent=self)
        self.__dict__['_hetero_store'] = {}

        mapping = {} if mapping is None else mapping
        for key, value in chain(mapping.items(), kwargs.items()):
            if '__' in key and isinstance(value, Mapping):
                key = tuple(key.split('__'))

            if isinstance(value, Mapping):
                self[key].update(value)
            else:
                setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        # `data.*` => Link to the `_global_store`.
        if key in self._global_store:
            return getattr(self._global_store, key)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, item: Any):
        # `data.* = ...` => Link to the `_global_store`.
        # NOTE: We aim to prevent duplicates in `_hetero_store` keys.
        if key in self._hetero_store:
            raise AttributeError(
                f"'{key}' attribute is already present as a node/edge-type")
        setattr(self._global_store, key, item)

    def __delattr__(self, key: str):
        # `del data.*` => Link to the `_global_store`.
        if key in self._global_store:
            delattr(self._global_store, key)

    def __getitem__(self, *args: Tuple[QueryType]) -> Any:
        # `data[*]` => Link to either `_hetero_store` or `_global_store`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.
        key = self._to_canonical(*args)

        out = self._hetero_store.get(key, None)
        if out is not None:
            return out

        out = self._global_store.get(key, None)
        if out is not None:
            return out

        if isinstance(key, tuple):
            out = EdgeStorage(_parent=self, _key=key)
        else:
            out = NodeStorage(_parent=self, _key=key)

        self._hetero_store[key] = out

        return out

    def __setitem__(self, key: str, item: Any):
        # `data[*] = ...` => Link to `data.* = ...` in `_global_store`.
        setattr(self, key, item)

    def __delitem__(self, *args: Tuple[QueryType]):
        # `del data[x]` => Link to either `_hetero_store` or `_global_store`.
        key = self._to_canonical(*args)
        if key in self._hetero_store:
            del self._hetero_store[key]
        elif key in self._global_store:
            del self._global_store[key]

    def __contains__(self, *args: Tuple[QueryType]) -> bool:
        key = self._to_canonical(*args)
        # TODO: Check child keys
        return key in self._global_store or key in self._hetero_store

    def __copy__(self):
        out = self.__class__()
        out.__dict__['_global_store'] = copy.copy(self._global_store)
        for key, item in self._hetero_store.items():
            out._hetero_store[key] = copy.copy(item)
            out._hetero_store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        out.__dict__['_global_store'] = copy.deepcopy(self._global_store, memo)
        for key, item in self._hetero_store.items():
            out._hetero_store[key] = copy.deepcopy(item, memo)
            out._hetero_store._parent = out
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        if self.is_homogeneous:
            info = [size_repr(k, v) for k, v in self._global_store.items()]
            return '{}({})'.format(cls, ', '.join(info))
        else:
            info1 = [size_repr(k, v, 2) for k, v in self._global_store.items()]
            info2 = [size_repr(k, v, 2) for k, v in self._hetero_store.items()]
            info = info1 + info2
            return '{}(\n{}\n)'.format(cls, ',\n'.join(info))

    def _to_canonical(
        self,
        *args: Tuple[QueryType],
    ) -> Union[NodeType, EdgeType]:
        # Converts a given `QueryType` to its "canonical type", i.e.
        # "incomplete" edge types `(src_node_type, dst_node_type)` will get
        # mapped to `(src_node_type, '_', dst_node_type)`.
        if len(args) == 1:
            args = args[0]
        if isinstance(args, tuple) and len(args) == 2:
            args = (args[0], '_', args[1])
        return args

    # Additional functionality ################################################

    def to_dict(self) -> Dict[str, Any]:
        out = self._global_store.to_dict()
        for key, value in self._hetero_store.items():
            out[key] = value.to_dict()
        return out

    def to_namedtuple(self) -> NamedTuple:
        field_names = list(self._global_store.keys())
        field_values = list(self._global_store.values())
        field_names += ['__'.join(key) for key in self._hetero_store.keys()]
        field_values += [
            store.to_namedtuple() for store in self._hetero_store.values()
        ]
        DataTuple = namedtuple('DataTuple', field_names)
        return DataTuple(*field_values)

    @property
    def is_heterogeneous(self) -> bool:
        return len(self._hetero_store) > 0

    @property
    def is_homogeneous(self) -> bool:
        return not self.is_heterogeneous

    @property
    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        # Returns the heterogeneous meta-data, i.e. its node and edge types.
        it = self._hetero_store.items()
        node_types = [k for k, v in it if isinstance(v, NodeStorage)]
        edge_types = [k for k, v in it if isinstance(v, EdgeStorage)]
        return node_types, edge_types

    @property
    def num_nodes(self) -> Optional[int]:
        if self.is_homogeneous:
            return self._global_store.num_nodes
        else:
            try:
                it = self._hetero_store.values()
                outs = [v.num_nodes for v in it if isinstance(v, NodeStorage)]
                return sum(outs)
            except TypeError:
                return None

    @property
    def num_edges(self) -> Optional[int]:
        if self.is_homogeneous:
            return self._global_store.num_edges
        else:
            try:
                it = self._hetero_store.values()
                outs = [v.num_edges for v in it if isinstance(v, EdgeStorage)]
                return sum(outs)
            except TypeError:
                return None

    @property
    @homogeneous_only
    def num_node_features(self) -> int:
        return self._global_store.num_node_features

    @property
    def num_features(self):
        return self.num_node_features

    @property
    @homogeneous_only
    def num_edge_features(self) -> int:
        return self._global_store.num_edge_features

    def is_coalesced(self) -> bool:
        raise NotImplementedError

    def coalesce(self):
        raise NotImplementedError

    def has_isolated_nodes(self) -> bool:
        raise NotImplementedError

    def has_self_loops(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def is_directed(self) -> bool:
        return not self.is_undirected

    def __cat_dim__(self, key: str,
                    value: Any) -> Optional[Union[int, Tuple[int, int]]]:
        # TODO: We only want to make use of this in homogeneous scenarios.
        if 'index' in key or 'face' in key:
            return -1
        elif key == 'adj_t' and isinstance(value, SparseTensor):
            return (0, 1)
        return 0

    def __inc__(self, key: str, value: Any) -> bool:
        # TODO: We only want to make use of this in homogeneous scenarios.
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def clone(self):
        return copy.deepcopy(self)

    def apply(self, func: Callable, *args: List[str]):
        self._global_store.apply(func, *args)
        for store in self._hetero_store.values():
            store.apply(func, *args)
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

    def debug(self):
        raise NotImplementedError

    # Deprecated functions ####################################################

    @classmethod
    @deprecated(details="use 'Data(dict)' instead")
    def from_dict(cls, dict: Dict[str, Any]):
        return cls(dict)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def keys(self) -> Iterable:
        out = list(self._global_store.keys())
        for store in self._hetero_store.values():
            out.append(list(store.keys()))
        return list(set(out))

    @property
    @deprecated(details="use 'data.face.size(-1)' instead")
    def num_faces(self) -> Optional[int]:
        if 'face' in self._global_store:
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    def __iter__(self):
        raise NotImplementedError

    def __call__(self, *keys):
        raise NotImplementedError

    @deprecated(details="use 'has_isolated_nodes' instead")
    def contains_isolated_nodes(self) -> bool:
        return self.has_isolated_nodes

    @deprecated(details="use 'has_self_loops' instead")
    def contains_self_loops(self) -> bool:
        return self.has_self_loops

    # End deprecated functions ################################################


def size_repr(key, value, indent=0):
    pad = ' ' * indent
    if isinstance(value, torch.Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, torch.Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 1:
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [pad + size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, Mapping):
        return f'{pad}\033[1m{key}\033[0m={out}'
    else:
        return f'{pad}{key}={out}'
