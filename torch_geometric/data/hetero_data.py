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
    r"""A data object describing a heterogeneous graph, holding multiple node
    and/or edge types in disjunct storage objects.
    Storage objects can hold either node-level, link-level or graph-level
    attributes.
    In general, :class:`~torch_geometric.data.HeteroData` tries to mimic the
    behaviour of a regular **nested** Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.

    .. code-block::

        from torch_geometric.data import HeteroData

        data = HeteroData()

        # Create two node types "paper" and "author" holding a feature matrix:
        data['paper'].x = torch.randn(num_papers, num_paper_features)
        data['author'].x = torch.randn(num_authors, num_authors_features)

        # Create an edge type "(author, writes, paper)" and building the
        # graph connectivity:
        data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]

        data['paper'].num_nodes
        >>> 23

        data['author', 'writes', 'paper'].num_edges
        >>> 52

        # PyTorch tensor functionality:
        data = data.pin_memory()
        data = data.to('cuda:0', non_blocking=True)

    Note that there exists multiple ways to create a heterogeneous graph data,
    *e.g.*:

    * To initialize a node of type :obj:`"paper"` holding a node feature
      matrix :obj:`x_paper` named :obj:`x`:

      .. code-block:: python

        from torch_geometric.data import HeteroData

        data = HeteroData()
        data['paper'].x = x_paper

        data = HeteroData(paper={ 'x': x_paper })

        data = HeteroData({'paper': { 'x': x_paper }})

    * To initialize an edge from source node type :obj:`"author"` to
      destination node type :obj:`"paper"` with relation type :obj:`"writes"`
      holding a graph connectivity matrix :obj:`edge_index_author_paper` named
      :obj:`edge_index`:

      .. code-block:: python

        data = HeteroData()
        data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

        data = HeteroData(author__writes__paper={
            'edge_index': edge_index_author_paper
        })

        data = HeteroData({
            ('author', 'writes', 'paper'):
            { 'edge_index': edge_index_author_paper }
        })
    """

    DEFAULT_REL = 'to'

    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        self._global_store = BaseStorage(_parent=self)
        self._node_store_dict = {}
        self._edge_store_dict = {}

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
        # `data[*]` => Link to either `_global_store`, _node_store_dict` or
        # `_edge_store_dict`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.
        key = self._to_canonical(*args)

        out = self._global_store.get(key, None)
        if out is not None:
            return out

        if isinstance(key, tuple):
            return self.get_edge_store(*key)
        else:
            return self.get_node_store(key)

    def __setitem__(self, key: str, value: Any):
        if key in chain(self.node_types, self.edge_types):
            raise AttributeError(
                f"'{key}' is already present as a node/edge-type")
        self._global_store[key] = value

    def __delitem__(self, *args: Tuple[QueryType]):
        # `del data[*]` => Link to `_node_store_dict` or `_edge_store_dict`.
        key = self._to_canonical(*args)
        if isinstance(key, tuple) and key in self.edge_types:
            del self._edge_store_dict[key]
        elif key in self.node_types:
            del self._node_store_dict[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._global_store = copy.copy(self._global_store)
        out._global_store._parent = out
        out._node_store_dict = {}
        for key, store in self._node_store_dict.items():
            out._node_store_dict[key] = copy.copy(store)
            out._node_store_dict[key]._parent = out
        out._edge_store_dict = {}
        for key, store in self._edge_store_dict.items():
            out._edge_store_dict[key] = copy.copy(store)
            out._edge_store_dict[key]._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key not in ['_node_store_dict', '_edge_store_dict']:
                out.__dict__[key] = copy.deepcopy(value, memo)
        out._global_store._parent = out
        out._node_store_dict = {}
        for key, store in self._node_store_dict.items():
            out._node_store_dict[key] = copy.deepcopy(store, memo)
            out._node_store_dict[key]._parent = out
        out._edge_store_dict = {}
        for key, store in self._edge_store_dict.items():
            out._edge_store_dict[key] = copy.deepcopy(store, memo)
            out._edge_store_dict[key]._parent = out
        return out

    def __repr__(self) -> str:
        info1 = [size_repr(k, v, 2) for k, v in self._global_store.items()]
        info2 = [size_repr(k, v, 2) for k, v in self._node_store_dict.items()]
        info3 = [size_repr(k, v, 2) for k, v in self._edge_store_dict.items()]
        info = ',\n'.join(info1 + info2 + info3)
        info = f'\n{info}\n' if len(info) > 0 else info
        return f'{self.__class__.__name__}({info})'

    @property
    def stores(self) -> List[BaseStorage]:
        r"""Returns a list of all storages of the graph."""
        return ([self._global_store] + list(self.node_stores) +
                list(self.edge_stores))

    @property
    def node_types(self) -> List[NodeType]:
        r"""Returns a list of all node types of the graph."""
        return list(self._node_store_dict.keys())

    @property
    def node_stores(self) -> List[NodeStorage]:
        r"""Returns a list of all node storages of the graph."""
        return list(self._node_store_dict.values())

    @property
    def edge_types(self) -> List[EdgeType]:
        r"""Returns a list of all edge types of the graph."""
        return list(self._edge_store_dict.keys())

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        r"""Returns a list of all edge storages of the graph."""
        return list(self._edge_store_dict.values())

    def to_dict(self) -> Dict[str, Any]:
        out = self._global_store.to_dict()
        for key, store in chain(self._node_store_dict.items(),
                                self._edge_store_dict.items()):
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

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph."""
        return super().num_nodes

    def debug(self):
        pass  # TODO

    ###########################################################################

    def _to_canonical(self, *args: Tuple[QueryType]) -> NodeOrEdgeType:
        # Converts a given `QueryType` to its "canonical type":
        # 1. `relation_type` will get mapped to the unique
        #    `(src_node_type, relation_type, dst_node_type)` tuple.
        # 2. `(src_node_type, dst_node_type)` will get mapped to the unique
        #    `(src_node_type, *, dst_node_type)` tuple, and
        #    `(src_node_type, 'to', dst_node_type)` otherwise.
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
                args = (args[0], self.DEFAULT_REL, args[1])

        return args

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        r"""Returns the heterogeneous meta-data, *i.e.* its node and edge
        types.

        .. code-block:: python

            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...
            data['author', 'writes', 'paper'].edge_index = ...

            print(data.metadata())
            >>> (['paper', 'author'], [('author', 'writes', 'paper')])
        """
        return self.node_types, self.edge_types

    def collect(self, key: str) -> Dict[NodeOrEdgeType, Any]:
        r"""Collects the attribute :attr:`key` from all node and edge types.

        .. code-block:: python

            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...

            print(data.collect('x'))
            >>> { 'paper': ..., 'author': ...}

        .. note::

            This is equivalent to writing :obj:`data.x_dict`.
        """
        mapping = {}
        for subtype, store in chain(self._node_store_dict.items(),
                                    self._edge_store_dict.items()):
            if key in store:
                mapping[subtype] = store[key]
        return mapping

    def get_node_store(self, key: NodeType) -> NodeStorage:
        r"""Gets the :class:`~torch_geometric.data.storage.NodeStorage` object
        of a particular node type :attr:`key`.
        If the storage is not present yet, will create a new
        :class:`torch_geometric.data.storage.NodeStorage` object for the given
        node type.

        .. code-block:: python

            data = HeteroData()
            node_storage = data.get_node_store('paper')
        """
        out = self._node_store_dict.get(key, None)
        if out is None:
            out = NodeStorage(_parent=self, _key=key)
            self._node_store_dict[key] = out
        return out

    def get_edge_store(self, src: str, rel: str, dst: str) -> EdgeStorage:
        r"""Gets the :class:`~torch_geometric.data.storage.EdgeStorage` object
        of a particular edge type given by the tuple :obj:`(src, rel, dst)`.
        If the storage is not present yet, will create a new
        :class:`torch_geometric.data.storage.EdgeStorage` object for the given
        edge type.

        .. code-block:: python

            data = HeteroData()
            edge_storage = data.get_edge_store('author', 'writes', 'paper')
        """
        key = (src, rel, dst)
        out = self._edge_store_dict.get(key, None)
        if out is None:
            out = EdgeStorage(_parent=self, _key=key)
            self._edge_store_dict[key] = out
        return out
