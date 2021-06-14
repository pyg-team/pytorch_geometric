from typing import Union, Tuple, List, Dict, Any, Optional, NamedTuple
from torch_geometric.typing import NodeType, EdgeType, QueryType

import copy
import re
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
    r"""A Python object modeling a single heterogeneous graph object inherits
    from :class:`torch_geometric.data.BaseData` type.

    There are a few ways to create a heterogeneous graph data.
    * To initialize a node of type `paper` with a feature Tensor `x_paper`
    named `x`:
    .. code-block:: python
      data = HeteroData()
      data['paper'].x = x_paper
      data = HeteroData(paper={'x': x_paper})
      data = HeteroData({'paper': {'x': x_paper}})
    * To initialize an edge from a node type `author` to another node type
    `paper` with edge index Tensor:
    .. code-block:: python
      data = HeteroData()
      data['author', 'writes', 'paper'].edge_index = edge_index_author_paper
      data = HeteroData(
        author_writes_paper={'edge_index': edge_index_author_paper)
      data = HeteroData({
        ('author', 'writes', 'paper'):
        {'edge_index': edge_index_author_paper}})
    """
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        self._global_store = BaseStorage(_parent=self)
        self._node_stores_dict = {}
        self._edge_stores_dict = {}

        for key, value in chain((_mapping or {}).items(), kwargs.items()):
            if '__' in key and isinstance(value, Mapping):
                key = tuple(key.split('__'))

            if isinstance(value, Mapping):
                self[key].update(value)
            else:
                setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        # `data.*_dict` => Link to node and edge stores.
        # `data.*` => Link to the `_global_store`.
        # It is the same as using `collect` to collect nodes and edges features
        # and use `attribute` to get graph attribute.
        if bool(re.search('_dict$', key)):
            out = self.collect(key[:-5])
            if len(out) > 0:
                return out
        return getattr(self._global_store, key)

    def __setattr__(self, key: str, value: Any):
        # `data._* = ...` => Link to the private `__dict__` store.
        # `data.* = ...` => Link to the `_global_store`.
        # NOTE: We aim to prevent duplicates in node or edge keys.
        if key[:1] == '_':
            self.__dict__[key] = value
        else:
            if key in self.node_types:
                raise AttributeError(
                    f"'{key}' is already present as a node type")
            elif key in self.edge_types:
                raise AttributeError(
                    f"'{key}' is already present as an edge type")
            setattr(self._global_store, key, value)

    def __delattr__(self, key: str):
        # `del data._*` => Link to the private `__dict__` store.
        # `del data.*` => Link to the `_global_store`.
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            delattr(self._global_store, key)

    def __getitem__(self, *args: Tuple[QueryType]) -> Any:
        # `data[*]` => Link to either `_global_store`, _node_stores_dict` or
        # `_edge_stores_dict`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.
        key = self._to_canonical(*args)

        out = self._global_store.get(key, None)
        if out is not None:
            return out

        if isinstance(key, tuple):
            return self.get_edges(key)
        else:
            return self.get_nodes(key)

    def __setitem__(self, key: str, value: Any):
        if key in chain(self.node_types, self.edge_types):
            raise AttributeError(
                f"'{key}' is already present as a node/edge-type")
        self._global_store[key] = value

    def __delitem__(self, *args: Tuple[QueryType]):
        # `del data[*]` => Link to `_node_stores_dict` or `_edge_stores_dict`.
        key = self._to_canonical(*args)
        if isinstance(key, tuple) and key in self.edge_types:
            del self._edge_stores_dict[key]
        elif key in self.node_types:
            del self._node_stores_dict[key]

    def __copy__(self):
        out = self.__class__()
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._global_store = copy.copy(self._global_store)
        out._global_store._parent = out
        out._node_stores_dict = {}
        for key, store in self._node_stores_dict.items():
            out._node_stores_dict[key] = copy.copy(store)
            out._node_stores_dict[key]._parent = out
        out._edge_stores_dict = {}
        for key, store in self._edge_stores_dict.items():
            out._edge_stores_dict[key] = copy.copy(store)
            out._edge_stores_dict[key]._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, value in self.__dict__.items():
            if key not in ['_node_stores_dict', '_edge_stores_dict']:
                out.__dict__[key] = copy.deepcopy(value, memo)
        out._global_store._parent = out
        out._node_stores_dict = {}
        for key, store in self._node_stores_dict.items():
            out._node_stores_dict[key] = copy.deepcopy(store, memo)
            out._node_stores_dict[key]._parent = out
        out._edge_stores_dict = {}
        for key, store in self._edge_stores_dict.items():
            out._edge_stores_dict[key] = copy.deepcopy(store, memo)
            out._edge_stores_dict[key]._parent = out
        return out

    def __repr__(self) -> str:
        info1 = [size_repr(k, v, 2) for k, v in self._global_store.items()]
        info2 = [size_repr(k, v, 2) for k, v in self._node_stores_dict.items()]
        info3 = [size_repr(k, v, 2) for k, v in self._edge_stores_dict.items()]
        info = info1 + info2 + info3
        return '{}(\n{}\n)'.format(self.__class__.__name__, ',\n'.join(info))

    def _all_nodes_and_edges(self):
        # Returns all node storage items and edge storage items.
        return chain(self._node_stores_dict.items(),
                     self._edge_stores_dict.items())

    @property
    def stores(self) -> List[BaseStorage]:
        # Return a list of all storages of the graph.
        return ([self._global_store] + list(self.node_stores) +
                list(self.edge_stores))

    @property
    def node_types(self) -> List[NodeType]:
        # Return a list of all node types of the graph.
        return list(self._node_stores_dict.keys())

    @property
    def node_stores(self) -> List[NodeStorage]:
        # Return a list of all node storages of the graph.
        return list(self._node_stores_dict.values())

    @property
    def edge_types(self) -> List[EdgeType]:
        # Return a list of all edge types of the graph.
        return list(self._edge_stores_dict.keys())

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        # Return a list of all edge storages of the graph.
        return list(self._edge_stores_dict.values())

    def to_dict(self) -> Dict[str, Any]:
        out = self._global_store.to_dict()
        for key, store in self._all_nodes_and_edges():
            out[key] = store.to_dict()
        return out

    def to_namedtuple(self) -> NamedTuple:
        field_names = list(self._global_store.keys())
        field_values = list(self._global_store.values())
        field_names += [
            '__'.join(key) if isinstance(key, tuple) else key
            for key in self.node_types + self.edge_types
        ]
        field_values += [
            store.to_namedtuple()
            for store in self.node_stores + self.edge_stores
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
        return self.node_types, self.edge_types

    def collect(self, key: str) -> Dict[NodeOrEdgeType, Any]:
        r'''Collects the attribute `key` from `_node_stores_dict` and
        `_edge_stores_dict`.
        '''
        mapping = {}
        for subtype, store in self._all_nodes_and_edges():
            if key in store:
                mapping[subtype] = store[key]
        return mapping

    def attribute(self, key: str) -> Any:
        # Get the attribute `key` from `_global_store`.
        return getattr(self._global_store, key)

    def get_nodes(self, key: NodeType) -> NodeStorage:
        r'''Get the storage of a particular node type. If it is not present, we
        create a new `Storage` object for the given node.
        Examples:
        ..code - block:: python
          data = HeteroData()
          paper = data.get_nodes('paper')
        '''
        out = self._node_stores_dict.get(key, None)
        if out is None:
            out = NodeStorage(_parent=self, _key=key)
            self._node_stores_dict[key] = out
        return out

    def get_edges(self, key: EdgeType) -> EdgeStorage:
        r'''Get the storage of a particular edge type. If it is not present, we
        create a new `Storage` object for the given edge.
        Examples:
        ..code - block:: python
          data = HeteroData()
          author_paper_edge = data.get_edges(('author', '_', 'paper'))
        '''
        out = self._edge_stores_dict.get(key, None)
        if out is None:
            out = EdgeStorage(_parent=self, _key=key)
            self._edge_stores_dict[key] = out
        return out
