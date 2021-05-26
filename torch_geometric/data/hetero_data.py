from typing import Union, Tuple, List, Dict, Any, Optional, NamedTuple
from torch_geometric.typing import NodeType, EdgeType, QueryType

import re
import copy
from itertools import chain
from collections import namedtuple
from collections.abc import Mapping

import torch
from torch_sparse import SparseTensor

from torch_geometric.data.data import BaseData, size_repr
from torch_geometric.data.storage import (BaseStorage, NodeStorage,
                                          EdgeStorage)

NodeOrEdgeType = Union[NodeType, EdgeType]
NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]


class HeteroData(BaseData):
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        self._global_store = BaseStorage(_parent=self)
        self._hetero_stores = {}

        for key, value in chain((_mapping or {}).items(), kwargs.items()):
            if '__' in key and isinstance(value, Mapping):
                key = tuple(key.split('__'))

            if isinstance(value, Mapping):
                self[key].update(value)
            else:
                setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        # `data.*_dict` => Link to the `_hetero_stores`.
        # `data.*` => Link to the `_global_store`.
        if bool(re.search('_dict$', key)):
            out = self.collect(key[:-5])
            if len(out) > 0:
                return out
        return getattr(self._global_store, key)

    def __setattr__(self, key: str, value: Any):
        # `data._* = ...` => Link to the private `__dict__` store.
        # `data.* = ...` => Link to the `_global_store`.
        # NOTE: We aim to prevent duplicates in `_hetero_store` keys.
        if key[:1] == '_':
            self.__dict__[key] = value
        else:
            if key in self._hetero_stores.keys():
                raise AttributeError(
                    f"'{key}' is already present as a node/edge-type")
            setattr(self._global_store, key, value)

    def __delattr__(self, key: str):
        # `del data._*` => Link to the private `__dict__` store.
        # `del data.*` => Link to the `_global_store`.
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            delattr(self._global_store, key)

    def __getitem__(self, *args: Tuple[QueryType]) -> Any:
        # `data[*]` => Link to either `_hetero_stores` or `_global_store`.
        # `data['*_dict']` => Link to the `_hetero_store`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.
        key = self._to_canonical(*args)

        out = self._hetero_stores.get(key, None)
        if out is not None:
            return out

        out = self._global_store.get(key, None)
        if out is not None:
            return out

        if isinstance(key, str) and bool(re.search('_dict$', key)):
            out = self.collect(key[:-5])
            if len(out) > 0:
                return out

        if isinstance(key, tuple):
            out = EdgeStorage(_parent=self, _key=key)
        else:
            out = NodeStorage(_parent=self, _key=key)

        self._hetero_stores[key] = out

        return out

    def __setitem__(self, key: str, value: Any):
        if key in self._hetero_stores.keys():
            raise AttributeError(
                f"'{key}' is already present as a node/edge-type")
        self._global_store[key] = value

    def __delitem__(self, *args: Tuple[QueryType]):
        # `del data[x]` => Link to either `_hetero_stores` or `_global_store`.
        key = self._to_canonical(*args)
        if key in self._hetero_stores.keys():
            del self._hetero_stores[key]
        elif key in self._global_store.keys():
            del self._global_store[key]

    def __copy__(self):
        out = self.__class__()
        for key, value in self.__dict__.items():
            if key not in ['_global_store', '_hetero_stores']:
                out.__dict__[key] = value
        out._global_store = copy.copy(self._global_store)
        out._global_store._parent = out
        out._hetero_stores = {}
        for key, store in self._hetero_stores.items():
            out._hetero_stores[key] = copy.copy(store)
            out._hetero_stores[key]._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, value in self.__dict__.items():
            if key not in ['_hetero_stores']:
                out.__dict__[key] = copy.deepcopy(value, memo)
        out._global_store._parent = out
        out._hetero_stores = {}
        for key, store in self._hetero_stores.items():
            out._hetero_stores[key] = copy.deepcopy(store, memo)
            out._hetero_stores[key]._parent = out
        return out

    def __repr__(self) -> str:
        info1 = [size_repr(k, v, 2) for k, v in self._global_store.items()]
        info2 = [size_repr(k, v, 2) for k, v in self._hetero_stores.items()]
        info = info1 + info2
        return '{}(\n{}\n)'.format(self.__class__.__name__, ',\n'.join(info))

    @property
    def _stores(self) -> List[BaseStorage]:
        return [self._global_store] + list(self._hetero_stores.values())

    @property
    def _node_stores(self) -> List[NodeStorage]:
        it = self._hetero_stores.values()
        return [store for store in it if isinstance(store, NodeStorage)]

    @property
    def _edge_stores(self) -> List[EdgeStorage]:
        it = self._hetero_stores.values()
        return [store for store in it if isinstance(store, EdgeStorage)]

    def to_dict(self) -> Dict[str, Any]:
        out = self._global_store.to_dict()
        for key, store in self._hetero_stores.items():
            out[key] = store.to_dict()
        return out

    def to_namedtuple(self) -> NamedTuple:
        field_names = list(self._global_store.keys())
        field_values = list(self._global_store.values())
        field_names += [
            '__'.join(key) if isinstance(key, tuple) else key
            for key in self._hetero_stores.keys()
        ]
        field_values += [
            store.to_namedtuple() for store in self._hetero_stores.values()
        ]
        DataTuple = namedtuple('DataTuple', field_names)
        return DataTuple(*field_values)

    def __cat_dim__(self, key: str, value: Any,
                    store: Optional[NodeOrEdgeStorage] = None) -> Any:
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        return 0

    def __inc__(self, key: str, value: Any,
                store: Optional[NodeOrEdgeStorage] = None) -> Any:
        if isinstance(store, EdgeStorage) and 'index' in key:
            return torch.tensor(store.size()).view(2, 1)
        else:
            return 0

    def debug(self):
        pass  # TODO

    ###########################################################################

    def _to_canonical(self, *args: Tuple[QueryType]) -> NodeOrEdgeType:
        # Converts a given `QueryType` to its "canonical type":
        # 1. `relation_type` will get mapped to the unique
        #    `(src_node_type, relation_type, dst_node_type)` tuple.
        # 2. `(src_node_type, dst_node_type)` will get mapped to the unique
        #    `(src_node_type, *, dst_node_type)` tuple, and
        #    `(src_node_type, '_', dst_node_type)` otherwise.
        if len(args) == 1:
            args = args[0]

        if isinstance(args, str):
            # Try to map to edge type based on unique relation type:
            edge_types = [key for key in self.metadata()[1] if key[1] == args]
            if len(edge_types) == 1:
                args = edge_types[0]

        elif len(args) == 2:
            # Try to find the unique source/destination node tuple:
            edge_types = [
                key for key in self.metadata()[1]
                if key[0] == args[0] and key[-1] == args[-1]
            ]
            if len(edge_types) == 1:
                args = edge_types[0]
            else:
                args = (args[0], '_', args[1])

        return args

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        # Returns the heterogeneous meta-data, i.e. its node and edge types.
        it = self._hetero_stores.items()
        node_types = [k for k, v in it if isinstance(v, NodeStorage)]
        edge_types = [k for k, v in it if isinstance(v, EdgeStorage)]
        return node_types, edge_types

    def collect(self, key: str) -> Dict[NodeOrEdgeType, Any]:
        # Collects the attribute `key` from all `_hetero_stores`.
        mapping = {}
        for subtype, store in self._hetero_stores.items():
            if key in store:
                mapping[subtype] = store[key]
        return mapping
