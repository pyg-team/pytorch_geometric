from typing import Union, Tuple, List, Dict, Any, Optional, NamedTuple
from torch_geometric.typing import NodeType, EdgeType, QueryType

import copy
import re
from itertools import chain
from collections.abc import Mapping
from collections import namedtuple, defaultdict

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data.data import BaseData, Data, size_repr
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
        self.__dict__['_global_store'] = BaseStorage(_parent=self)
        self.__dict__['_node_store_dict'] = {}
        self.__dict__['_edge_store_dict'] = {}

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
        # Using `data.*_dict` is the same as using `collect()` for collecting
        # nodes and edges features.
        if bool(re.search('_dict$', key)):
            out = self.collect(key[:-5])
            if len(out) > 0:
                return out
        return getattr(self._global_store, key)

    def __setattr__(self, key: str, value: Any):
        # NOTE: We aim to prevent duplicates in node or edge types.
        if key in self.node_types:
            raise AttributeError(f"'{key}' is already present as a node type")
        elif key in self.edge_types:
            raise AttributeError(f"'{key}' is already present as an edge type")
        setattr(self._global_store, key, value)

    def __delattr__(self, key: str):
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
        if key in self.node_types:
            raise AttributeError(f"'{key}' is already present as a node type")
        elif key in self.edge_types:
            raise AttributeError(f"'{key}' is already present as an edge type")
        self._global_store[key] = value

    def __delitem__(self, *args: Tuple[QueryType]):
        # `del data[*]` => Link to `_node_store_dict` or `_edge_store_dict`.
        key = self._to_canonical(*args)
        if key in self.edge_types:
            del self._edge_store_dict[key]
        elif key in self.node_types:
            del self._node_store_dict[key]
        else:
            del self._global_store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_global_store'] = copy.copy(self._global_store)
        out._global_store._parent = out
        out.__dict__['_node_store_dict'] = {}
        for key, store in self._node_store_dict.items():
            out._node_store_dict[key] = copy.copy(store)
            out._node_store_dict[key]._parent = out
        out.__dict__['_edge_store_dict'] = {}
        for key, store in self._edge_store_dict.items():
            out._edge_store_dict[key] = copy.copy(store)
            out._edge_store_dict[key]._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._global_store._parent = out
        for key in self._node_store_dict.keys():
            out._node_store_dict[key]._parent = out
        for key in out._edge_store_dict.keys():
            out._edge_store_dict[key]._parent = out
        return out

    def __repr__(self) -> str:
        info1 = [size_repr(k, v, 2) for k, v in self._global_store.items()]
        info2 = [size_repr(k, v, 2) for k, v in self._node_store_dict.items()]
        info3 = [size_repr(k, v, 2) for k, v in self._edge_store_dict.items()]
        info = ',\n'.join(info1 + info2 + info3)
        info = f'\n{info}\n' if len(info) > 0 else info
        return f'{self.__class__.__name__}({info})'

    def stores_as(self, data: 'HeteroData'):
        for node_type in data.node_types:
            self.get_node_store(node_type)
        for edge_type in data.edge_types:
            self.get_edge_store(*edge_type)
        return self

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
                    store: Optional[NodeOrEdgeStorage] = None, *args,
                    **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        return 0

    def __inc__(self, key: str, value: Any,
                store: Optional[NodeOrEdgeStorage] = None, *args,
                **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif isinstance(store, EdgeStorage) and 'index' in key:
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
            node_types = [key for key in self.node_types if key == args]
            if len(node_types) == 1:
                args = node_types[0]
                return args

            # Try to map to edge type based on unique relation type:
            edge_types = [key for key in self.edge_types if key[1] == args]
            if len(edge_types) == 1:
                args = edge_types[0]
                return args

        if len(args) == 2:
            # Try to find the unique source/destination node tuple:
            edge_types = [
                key for key in self.edge_types
                if key[0] == args[0] and key[-1] == args[-1]
            ]
            if len(edge_types) == 1:
                args = edge_types[0]
                return args
            else:
                args = (args[0], self.DEFAULT_REL, args[1])
                return args

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
            if hasattr(store, key):
                mapping[subtype] = getattr(store, key)
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

    def to_homogeneous(self, node_attrs: Optional[List[str]] = None,
                       edge_attrs: Optional[List[str]] = None,
                       add_node_type: bool = True,
                       add_edge_type: bool = True) -> Data:
        """Converts a :class:`~torch_geometric.data.HeteroData` object to a
        homogeneous :class:`~torch_geometric.data.Data` object.
        By default, all features with same feature dimensionality across
        different types will be merged into a single representation, unless
        otherwise specified via the :obj:`node_attrs` and :obj:`edge_attrs`
        arguments.
        Furthermore, attributes named :obj:`node_type` and :obj:`edge_type`
        will be added to the returned :class:`~torch_geometric.data.Data`
        object, denoting node-level and edge-level vectors holding the
        node and edge type as integers, respectively.

        Args:
            node_attrs (List[str], optional): The node features to combine
                across all node types. These node features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which node features to combine.
                (default: :obj:`None`)
            edge_attrs (List[str], optional): The edge features to combine
                across all edge types. These edge features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which edge features to combine.
                (default: :obj:`None`)
            add_node_type (bool, optional): If set to :obj:`False`, will not
                add the node-level vector :obj:`node_type` to the returned
                :class:`~torch_geometric.data.Data` object.
                (default: :obj:`True`)
            add_edge_type (bool, optional): If set to :obj:`False`, will not
                add the edge-level vector :obj:`edge_type` to the returned
                :class:`~torch_geometric.data.Data` object.
                (default: :obj:`True`)
        """
        def _consistent_size(stores: List[BaseStorage]) -> List[str]:
            sizes_dict = defaultdict(list)
            for store in stores:
                for key, value in store.items():
                    if key in ['edge_index', 'adj_t']:
                        continue
                    if isinstance(value, Tensor):
                        dim = self.__cat_dim__(key, value, store)
                        size = value.size()[:dim] + value.size()[dim + 1:]
                        sizes_dict[key].append(tuple(size))
            return [
                k for k, sizes in sizes_dict.items()
                if len(sizes) == len(stores) and len(set(sizes)) == 1
            ]

        data = Data(**self._global_store.to_dict())

        # Iterate over all node stores and record the slice information:
        node_slices, cumsum = {}, 0
        node_type_names, node_types = [], []
        for i, (node_type, store) in enumerate(self._node_store_dict.items()):
            num_nodes = store.num_nodes
            node_slices[node_type] = (cumsum, cumsum + num_nodes)
            node_type_names.append(node_type)
            cumsum += num_nodes

            if add_node_type:
                kwargs = {'dtype': torch.long}
                node_types.append(torch.full((num_nodes, ), i, **kwargs))
        data._node_type_names = node_type_names

        if len(node_types) > 1:
            data.node_type = torch.cat(node_types, dim=0)
        elif len(node_types) == 1:
            data.node_type = node_types[0]

        # Combine node attributes into a single tensor:
        if node_attrs is None:
            node_attrs = _consistent_size(self.node_stores)
        for key in node_attrs:
            values = [store[key] for store in self.node_stores]
            dim = self.__cat_dim__(key, values[0], self.node_stores[0])
            value = torch.cat(values, dim) if len(values) > 1 else values[0]
            data[key] = value

        if len([
                key for key in node_attrs
                if (key in {'x', 'pos', 'batch'} or 'node' in key)
        ]) == 0 and not add_node_type:
            data.num_nodes = cumsum

        # Iterate over all edge stores and record the slice information:
        edge_slices, cumsum = {}, 0
        edge_indices, edge_type_names, edge_types = [], [], []
        for i, (edge_type, store) in enumerate(self._edge_store_dict.items()):
            src, _, dst = edge_type
            num_edges = store.num_edges
            edge_slices[edge_type] = (cumsum, cumsum + num_edges)
            edge_type_names.append(edge_type)
            cumsum += num_edges

            kwargs = {'dtype': torch.long, 'device': store.edge_index.device}
            offset = [[node_slices[src][0]], [node_slices[dst][0]]]
            offset = torch.tensor(offset, **kwargs)
            edge_indices.append(store.edge_index + offset)
            if add_edge_type:
                edge_types.append(torch.full((num_edges, ), i, **kwargs))
        data._edge_type_names = edge_type_names

        if len(edge_indices) > 1:
            data.edge_index = torch.cat(edge_indices, dim=-1)
        elif len(edge_indices) == 1:
            data.edge_index = edge_indices[0]

        if len(edge_types) > 1:
            data.edge_type = torch.cat(edge_types, dim=0)
        elif len(edge_types) == 1:
            data.edge_type = edge_types[0]

        # Combine edge attributes into a single tensor:
        if edge_attrs is None:
            edge_attrs = _consistent_size(self.edge_stores)
        for key in edge_attrs:
            values = [store[key] for store in self.edge_stores]
            dim = self.__cat_dim__(key, values[0], self.edge_stores[0])
            value = torch.cat(values, dim) if len(values) > 1 else values[0]
            data[key] = value

        return data
