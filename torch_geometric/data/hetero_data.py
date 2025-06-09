import copy
import re
import warnings
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Self

from torch_geometric import Index
from torch_geometric.data import EdgeAttr, FeatureStore, GraphStore, TensorAttr
from torch_geometric.data.data import BaseData, Data, size_repr, warn_or_raise
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage
from torch_geometric.typing import (
    DEFAULT_REL,
    EdgeTensorType,
    EdgeType,
    FeatureTensorType,
    NodeOrEdgeType,
    NodeType,
    QueryType,
    SparseTensor,
    TensorFrame,
    torch_frame,
)
from torch_geometric.utils import (
    bipartite_subgraph,
    contains_isolated_nodes,
    is_sparse,
    is_undirected,
    mask_select,
)

NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

_DISPLAYED_TYPE_NAME_WARNING: bool = False


class HeteroData(BaseData, FeatureStore, GraphStore):
    r"""A data object describing a heterogeneous graph, holding multiple node
    and/or edge types in disjunct storage objects.
    Storage objects can hold either node-level, link-level or graph-level
    attributes.
    In general, :class:`~torch_geometric.data.HeteroData` tries to mimic the
    behavior of a regular **nested** :python:`Python` dictionary.
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

        # (1) Assign attributes after initialization,
        data = HeteroData()
        data['paper'].x = x_paper

        # or (2) pass them as keyword arguments during initialization,
        data = HeteroData(paper={ 'x': x_paper })

        # or (3) pass them as dictionaries during initialization,
        data = HeteroData({'paper': { 'x': x_paper }})

    * To initialize an edge from source node type :obj:`"author"` to
      destination node type :obj:`"paper"` with relation type :obj:`"writes"`
      holding a graph connectivity matrix :obj:`edge_index_author_paper` named
      :obj:`edge_index`:

      .. code-block:: python

        # (1) Assign attributes after initialization,
        data = HeteroData()
        data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

        # or (2) pass them as keyword arguments during initialization,
        data = HeteroData(author__writes__paper={
            'edge_index': edge_index_author_paper
        })

        # or (3) pass them as dictionaries during initialization,
        data = HeteroData({
            ('author', 'writes', 'paper'):
            { 'edge_index': edge_index_author_paper }
        })
    """
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()

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

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> Self:
        r"""Creates a :class:`~torch_geometric.data.HeteroData` object from a
        dictionary.
        """
        out = cls()
        for key, value in mapping.items():
            if key == '_global_store':
                out.__dict__['_global_store'] = BaseStorage(
                    _parent=out, **value)
            elif isinstance(key, str):
                out._node_store_dict[key] = NodeStorage(
                    _parent=out, _key=key, **value)
            else:
                out._edge_store_dict[key] = EdgeStorage(
                    _parent=out, _key=key, **value)
        return out

    def __getattr__(self, key: str) -> Any:
        # `data.*_dict` => Link to node and edge stores.
        # `data.*` => Link to the `_global_store`.
        # Using `data.*_dict` is the same as using `collect()` for collecting
        # nodes and edges features.
        if hasattr(self._global_store, key):
            return getattr(self._global_store, key)
        elif bool(re.search('_dict$', key)):
            return self.collect(key[:-5])
        raise AttributeError(f"'{self.__class__.__name__}' has no "
                             f"attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        # NOTE: We aim to prevent duplicates in node or edge types.
        if key in self.node_types:
            raise AttributeError(f"'{key}' is already present as a node type")
        elif key in self.edge_types:
            raise AttributeError(f"'{key}' is already present as an edge type")
        setattr(self._global_store, key, value)

    def __delattr__(self, key: str):
        delattr(self._global_store, key)

    def __getitem__(self, *args: QueryType) -> Any:
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

    def __delitem__(self, *args: QueryType):
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

    def stores_as(self, data: Self):
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

    def node_items(self) -> List[Tuple[NodeType, NodeStorage]]:
        r"""Returns a list of node type and node storage pairs."""
        return list(self._node_store_dict.items())

    def edge_items(self) -> List[Tuple[EdgeType, EdgeStorage]]:
        r"""Returns a list of edge type and edge storage pairs."""
        return list(self._edge_store_dict.items())

    @property
    def input_type(self) -> Optional[Union[NodeType, EdgeType]]:
        r"""Returns the seed/input node/edge type of the graph in case it
        refers to a sampled subgraph, *e.g.*, obtained via
        :class:`~torch_geometric.loader.NeighborLoader` or
        :class:`~torch_geometric.loader.LinkNeighborLoader`.
        """
        for node_type, store in self.node_items():
            if hasattr(store, 'input_id'):
                return node_type
        for edge_type, store in self.edge_items():
            if hasattr(store, 'input_id'):
                return edge_type
        return None

    def to_dict(self) -> Dict[str, Any]:
        out_dict: Dict[str, Any] = {}
        out_dict['_global_store'] = self._global_store.to_dict()
        for key, store in chain(self._node_store_dict.items(),
                                self._edge_store_dict.items()):
            out_dict[key] = store.to_dict()
        return out_dict

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

    def set_value_dict(
        self,
        key: str,
        value_dict: Dict[str, Any],
    ) -> Self:
        r"""Sets the values in the dictionary :obj:`value_dict` to the
        attribute with name :obj:`key` to all node/edge types present in the
        dictionary.

        .. code-block:: python

           data = HeteroData()

           data.set_value_dict('x', {
               'paper': torch.randn(4, 16),
               'author': torch.randn(8, 32),
           })

           print(data['paper'].x)
        """
        for k, v in (value_dict or {}).items():
            self[k][key] = v
        return self

    def update(self, data: Self) -> Self:
        for store in data.stores:
            for key, value in store.items():
                self[store._key][key] = value
        return self

    def __cat_dim__(self, key: str, value: Any,
                    store: Optional[NodeOrEdgeStorage] = None, *args,
                    **kwargs) -> Any:
        if is_sparse(value) and ('adj' in key or 'edge_index' in key):
            return (0, 1)
        elif isinstance(store, EdgeStorage) and 'index' in key:
            return -1
        return 0

    def __inc__(self, key: str, value: Any,
                store: Optional[NodeOrEdgeStorage] = None, *args,
                **kwargs) -> Any:
        if 'batch' in key and isinstance(value, Tensor):
            if isinstance(value, Index):
                return value.get_dim_size()
            return int(value.max()) + 1
        elif isinstance(store, EdgeStorage) and 'index' in key:
            return torch.tensor(store.size()).view(2, 1)
        else:
            return 0

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph."""
        return super().num_nodes

    @property
    def num_node_features(self) -> Dict[NodeType, int]:
        r"""Returns the number of features per node type in the graph."""
        return {
            key: store.num_node_features
            for key, store in self._node_store_dict.items()
        }

    @property
    def num_features(self) -> Dict[NodeType, int]:
        r"""Returns the number of features per node type in the graph.
        Alias for :py:attr:`~num_node_features`.
        """
        return self.num_node_features

    @property
    def num_edge_features(self) -> Dict[EdgeType, int]:
        r"""Returns the number of features per edge type in the graph."""
        return {
            key: store.num_edge_features
            for key, store in self._edge_store_dict.items()
        }

    def has_isolated_nodes(self) -> bool:
        r"""Returns :obj:`True` if the graph contains isolated nodes."""
        edge_index, _, _ = to_homogeneous_edge_index(self)
        return contains_isolated_nodes(edge_index, num_nodes=self.num_nodes)

    def is_undirected(self) -> bool:
        r"""Returns :obj:`True` if graph edges are undirected."""
        edge_index, _, _ = to_homogeneous_edge_index(self)
        return is_undirected(edge_index, num_nodes=self.num_nodes)

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        node_types = set(self.node_types)
        num_src_node_types = {src for src, _, _ in self.edge_types}
        num_dst_node_types = {dst for _, _, dst in self.edge_types}

        dangling_types = (num_src_node_types | num_dst_node_types) - node_types
        if len(dangling_types) > 0:
            status = False
            warn_or_raise(
                f"The node types {dangling_types} are referenced in edge "
                f"types but do not exist as node types", raise_on_error)

        dangling_types = node_types - (num_src_node_types | num_dst_node_types)
        if len(dangling_types) > 0:
            warn_or_raise(  # May be intended.
                f"The node types {dangling_types} are isolated and are not "
                f"referenced by any edge type ", raise_on_error=False)

        for edge_type, store in self._edge_store_dict.items():
            src, _, dst = edge_type

            num_src_nodes = self[src].num_nodes
            num_dst_nodes = self[dst].num_nodes
            if num_src_nodes is None:
                status = False
                warn_or_raise(
                    f"'num_nodes' is undefined in node type '{src}' of "
                    f"'{cls_name}'", raise_on_error)

            if num_dst_nodes is None:
                status = False
                warn_or_raise(
                    f"'num_nodes' is undefined in node type '{dst}' of "
                    f"'{cls_name}'", raise_on_error)

            if 'edge_index' in store:
                if (store.edge_index.dim() != 2
                        or store.edge_index.size(0) != 2):
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} needs to be "
                        f"of shape [2, num_edges] in '{cls_name}' (found "
                        f"{store.edge_index.size()})", raise_on_error)

            if 'edge_index' in store and store.edge_index.numel() > 0:
                if store.edge_index.min() < 0:
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains "
                        f"negative indices in '{cls_name}' "
                        f"(found {int(store.edge_index.min())})",
                        raise_on_error)

                if (num_src_nodes is not None
                        and store.edge_index[0].max() >= num_src_nodes):
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains "
                        f"larger source indices than the number of nodes "
                        f"({num_src_nodes}) of this node type in '{cls_name}' "
                        f"(found {int(store.edge_index[0].max())})",
                        raise_on_error)

                if (num_dst_nodes is not None
                        and store.edge_index[1].max() >= num_dst_nodes):
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains "
                        f"larger destination indices than the number of nodes "
                        f"({num_dst_nodes}) of this node type in '{cls_name}' "
                        f"(found {int(store.edge_index[1].max())})",
                        raise_on_error)

        return status

    def debug(self):
        pass  # TODO

    ###########################################################################

    def _to_canonical(self, *args: QueryType) -> NodeOrEdgeType:
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

        elif len(args) == 2:
            # Try to find the unique source/destination node tuple:
            edge_types = [
                key for key in self.edge_types
                if key[0] == args[0] and key[-1] == args[-1]
            ]
            if len(edge_types) == 1:
                args = edge_types[0]
                return args
            elif len(edge_types) == 0:
                args = (args[0], DEFAULT_REL, args[1])
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

    def collect(
        self,
        key: str,
        allow_empty: bool = False,
    ) -> Dict[NodeOrEdgeType, Any]:
        r"""Collects the attribute :attr:`key` from all node and edge types.

        .. code-block:: python

            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...

            print(data.collect('x'))
            >>> { 'paper': ..., 'author': ...}

        .. note::

            This is equivalent to writing :obj:`data.x_dict`.

        Args:
            key (str): The attribute to collect from all node and edge types.
            allow_empty (bool, optional): If set to :obj:`True`, will not raise
                an error in case the attribute does not exit in any node or
                edge type. (default: :obj:`False`)
        """
        mapping = {}
        for subtype, store in chain(self._node_store_dict.items(),
                                    self._edge_store_dict.items()):
            if hasattr(store, key):
                mapping[subtype] = getattr(store, key)
        if not allow_empty and len(mapping) == 0:
            raise KeyError(f"Tried to collect '{key}' but did not find any "
                           f"occurrences of it in any node and/or edge type")
        return mapping

    def _check_type_name(self, name: str):
        global _DISPLAYED_TYPE_NAME_WARNING
        if not _DISPLAYED_TYPE_NAME_WARNING and '__' in name:
            _DISPLAYED_TYPE_NAME_WARNING = True
            warnings.warn(
                f"There exist type names in the "
                f"'{self.__class__.__name__}' object that contain "
                f"double underscores '__' (e.g., '{name}'). This "
                f"may lead to unexpected behavior. To avoid any "
                f"issues, ensure that your type names only contain "
                f"single underscores.", stacklevel=2)

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
            self._check_type_name(key)
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
            self._check_type_name(rel)
            out = EdgeStorage(_parent=self, _key=key)
            self._edge_store_dict[key] = out
        return out

    def rename(self, name: NodeType, new_name: NodeType) -> Self:
        r"""Renames the node type :obj:`name` to :obj:`new_name` in-place."""
        node_store = self._node_store_dict.pop(name)
        node_store._key = new_name
        self._node_store_dict[new_name] = node_store

        for edge_type in self.edge_types:
            src, rel, dst = edge_type
            if src == name or dst == name:
                edge_store = self._edge_store_dict.pop(edge_type)
                src = new_name if src == name else src
                dst = new_name if dst == name else dst
                edge_type = (src, rel, dst)
                edge_store._key = edge_type
                self._edge_store_dict[edge_type] = edge_store

        return self

    def subgraph(self, subset_dict: Dict[NodeType, Tensor]) -> Self:
        r"""Returns the induced subgraph containing the node types and
        corresponding nodes in :obj:`subset_dict`.

        If a node type is not a key in :obj:`subset_dict` then all nodes of
        that type remain in the graph.

        .. code-block:: python

            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...
            data['conference'].x = ...
            data['paper', 'cites', 'paper'].edge_index = ...
            data['author', 'paper'].edge_index = ...
            data['paper', 'conference'].edge_index = ...
            print(data)
            >>> HeteroData(
                paper={ x=[10, 16] },
                author={ x=[5, 32] },
                conference={ x=[5, 8] },
                (paper, cites, paper)={ edge_index=[2, 50] },
                (author, to, paper)={ edge_index=[2, 30] },
                (paper, to, conference)={ edge_index=[2, 25] }
            )

            subset_dict = {
                'paper': torch.tensor([3, 4, 5, 6]),
                'author': torch.tensor([0, 2]),
            }

            print(data.subgraph(subset_dict))
            >>> HeteroData(
                paper={ x=[4, 16] },
                author={ x=[2, 32] },
                conference={ x=[5, 8] },
                (paper, cites, paper)={ edge_index=[2, 24] },
                (author, to, paper)={ edge_index=[2, 5] },
                (paper, to, conference)={ edge_index=[2, 10] }
            )

        Args:
            subset_dict (Dict[str, LongTensor or BoolTensor]): A dictionary
                holding the nodes to keep for each node type.
        """
        data = copy.copy(self)
        subset_dict = copy.copy(subset_dict)

        for node_type, subset in subset_dict.items():
            for key, value in self[node_type].items():
                if key == 'num_nodes':
                    if subset.dtype == torch.bool:
                        data[node_type].num_nodes = int(subset.sum())
                    else:
                        data[node_type].num_nodes = subset.size(0)
                elif self[node_type].is_node_attr(key):
                    data[node_type][key] = value[subset]
                else:
                    data[node_type][key] = value

        for edge_type in self.edge_types:
            if 'edge_index' not in self[edge_type]:
                continue

            src, _, dst = edge_type

            src_subset = subset_dict.get(src)
            if src_subset is None:
                src_subset = torch.arange(data[src].num_nodes)
            dst_subset = subset_dict.get(dst)
            if dst_subset is None:
                dst_subset = torch.arange(data[dst].num_nodes)

            edge_index, _, edge_mask = bipartite_subgraph(
                (src_subset, dst_subset),
                self[edge_type].edge_index,
                relabel_nodes=True,
                size=(self[src].num_nodes, self[dst].num_nodes),
                return_edge_mask=True,
            )

            for key, value in self[edge_type].items():
                if key == 'edge_index':
                    data[edge_type].edge_index = edge_index
                elif self[edge_type].is_edge_attr(key):
                    data[edge_type][key] = value[edge_mask]
                else:
                    data[edge_type][key] = value

        return data

    def edge_subgraph(
        self,
        subset_dict: Dict[EdgeType, Tensor],
    ) -> Self:
        r"""Returns the induced subgraph given by the edge indices in
        :obj:`subset_dict` for certain edge types.
        Will currently preserve all the nodes in the graph, even if they are
        isolated after subgraph computation.

        Args:
            subset_dict (Dict[Tuple[str, str, str], LongTensor or BoolTensor]):
                A dictionary holding the edges to keep for each edge type.
        """
        data = copy.copy(self)

        for edge_type, subset in subset_dict.items():
            edge_store, new_edge_store = self[edge_type], data[edge_type]
            for key, value in edge_store.items():
                if edge_store.is_edge_attr(key):
                    dim = self.__cat_dim__(key, value, edge_store)
                    if subset.dtype == torch.bool:
                        new_edge_store[key] = mask_select(value, dim, subset)
                    else:
                        new_edge_store[key] = value.index_select(dim, subset)

        return data

    def node_type_subgraph(self, node_types: List[NodeType]) -> Self:
        r"""Returns the subgraph induced by the given :obj:`node_types`, *i.e.*
        the returned :class:`HeteroData` object only contains the node types
        which are included in :obj:`node_types`, and only contains the edge
        types where both end points are included in :obj:`node_types`.
        """
        data = copy.copy(self)
        for edge_type in self.edge_types:
            src, _, dst = edge_type
            if src not in node_types or dst not in node_types:
                del data[edge_type]
        for node_type in self.node_types:
            if node_type not in node_types:
                del data[node_type]
        return data

    def edge_type_subgraph(self, edge_types: List[EdgeType]) -> Self:
        r"""Returns the subgraph induced by the given :obj:`edge_types`, *i.e.*
        the returned :class:`HeteroData` object only contains the edge types
        which are included in :obj:`edge_types`, and only contains the node
        types of the end points which are included in :obj:`node_types`.
        """
        edge_types = [self._to_canonical(e) for e in edge_types]

        data = copy.copy(self)
        for edge_type in self.edge_types:
            if edge_type not in edge_types:
                del data[edge_type]
        node_types = {e[0] for e in edge_types}
        node_types |= {e[-1] for e in edge_types}
        for node_type in self.node_types:
            if node_type not in node_types:
                del data[node_type]
        return data

    def to_homogeneous(
        self,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        add_node_type: bool = True,
        add_edge_type: bool = True,
        dummy_values: bool = True,
    ) -> Data:
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
            dummy_values (bool, optional): If set to :obj:`True`, will fill
                attributes of remaining types with dummy values.
                Dummy values are :obj:`NaN` for floating point attributes,
                :obj:`False` for booleans, and :obj:`-1` for integers.
                (default: :obj:`True`)
        """
        def get_sizes(stores: List[BaseStorage]) -> Dict[str, List[Tuple]]:
            sizes_dict = defaultdict(list)
            for store in stores:
                for key, value in store.items():
                    if key in [
                            'edge_index', 'edge_label_index', 'adj', 'adj_t'
                    ]:
                        continue
                    if isinstance(value, Tensor):
                        dim = self.__cat_dim__(key, value, store)
                        size = value.size()[:dim] + value.size()[dim + 1:]
                        sizes_dict[key].append(tuple(size))
            return sizes_dict

        def fill_dummy_(stores: List[BaseStorage],
                        keys: Optional[List[str]] = None):
            sizes_dict = get_sizes(stores)

            if keys is not None:
                sizes_dict = {
                    key: sizes
                    for key, sizes in sizes_dict.items() if key in keys
                }

            sizes_dict = {
                key: sizes
                for key, sizes in sizes_dict.items() if len(set(sizes)) == 1
            }

            for store in stores:  # Fill stores with dummy features:
                for key, sizes in sizes_dict.items():
                    if key not in store:
                        ref = list(self.collect(key).values())[0]
                        dim = self.__cat_dim__(key, ref, store)
                        if ref.is_floating_point():
                            dummy = float('NaN')
                        elif ref.dtype == torch.bool:
                            dummy = False
                        else:
                            dummy = -1
                        if isinstance(store, NodeStorage):
                            dim_size = store.num_nodes
                        else:
                            dim_size = store.num_edges
                        shape = sizes[0][:dim] + (dim_size, ) + sizes[0][dim:]
                        store[key] = torch.full(shape, dummy, dtype=ref.dtype,
                                                device=ref.device)

        def _consistent_size(stores: List[BaseStorage]) -> List[str]:
            sizes_dict = get_sizes(stores)
            keys = []
            for key, sizes in sizes_dict.items():
                # The attribute needs to exist in all types:
                if len(sizes) != len(stores):
                    continue
                # The attributes needs to have the same number of dimensions:
                lengths = {len(size) for size in sizes}
                if len(lengths) != 1:
                    continue
                # The attributes needs to have the same size in all dimensions:
                if len(sizes[0]) != 1 and len(set(sizes)) != 1:
                    continue
                keys.append(key)

            # Check for consistent column names in `TensorFrame`:
            tf_cols = defaultdict(list)
            for store in stores:
                for key, value in store.items():
                    if isinstance(value, TensorFrame):
                        cols = tuple(chain(*value.col_names_dict.values()))
                        tf_cols[key].append(cols)

            for key, cols in tf_cols.items():
                # The attribute needs to exist in all types:
                if len(cols) != len(stores):
                    continue
                # The attributes needs to have the same column names:
                lengths = set(cols)
                if len(lengths) != 1:
                    continue
                keys.append(key)

            return keys

        if dummy_values:
            self = copy.copy(self)
            fill_dummy_(self.node_stores, node_attrs)
            fill_dummy_(self.edge_stores, edge_attrs)

        edge_index, node_slices, edge_slices = to_homogeneous_edge_index(self)
        device = edge_index.device if edge_index is not None else None

        data = Data(**self._global_store.to_dict())
        if edge_index is not None:
            data.edge_index = edge_index
        data._node_type_names = list(node_slices.keys())
        data._edge_type_names = list(edge_slices.keys())

        # Combine node attributes into a single tensor:
        if node_attrs is None:
            node_attrs = _consistent_size(self.node_stores)
        for key in node_attrs:
            if key in {'ptr'}:
                continue
            values = [store[key] for store in self.node_stores]
            if isinstance(values[0], TensorFrame):
                value = torch_frame.cat(values, dim=0)
            else:
                dim = self.__cat_dim__(key, values[0], self.node_stores[0])
                dim = values[0].dim() + dim if dim < 0 else dim
                # For two-dimensional features, we allow arbitrary shapes and
                # pad them with zeros if necessary in case their size doesn't
                # match:
                if values[0].dim() == 2 and dim == 0:
                    _max = max([value.size(-1) for value in values])
                    for i, v in enumerate(values):
                        if v.size(-1) < _max:
                            pad = v.new_zeros(v.size(0), _max - v.size(-1))
                            values[i] = torch.cat([v, pad], dim=-1)
                value = torch.cat(values, dim)
            data[key] = value

        if not data.can_infer_num_nodes:
            data.num_nodes = list(node_slices.values())[-1][1]

        # Combine edge attributes into a single tensor:
        if edge_attrs is None:
            edge_attrs = _consistent_size(self.edge_stores)
        for key in edge_attrs:
            values = [store[key] for store in self.edge_stores]
            dim = self.__cat_dim__(key, values[0], self.edge_stores[0])
            value = torch.cat(values, dim) if len(values) > 1 else values[0]
            data[key] = value

        if 'edge_label_index' in self:
            edge_label_index_dict = self.edge_label_index_dict
            for edge_type, edge_label_index in edge_label_index_dict.items():
                edge_label_index = edge_label_index.clone()
                edge_label_index[0] += node_slices[edge_type[0]][0]
                edge_label_index[1] += node_slices[edge_type[-1]][0]
                edge_label_index_dict[edge_type] = edge_label_index
            data.edge_label_index = torch.cat(
                list(edge_label_index_dict.values()), dim=-1)

        if add_node_type:
            sizes = [offset[1] - offset[0] for offset in node_slices.values()]
            sizes = torch.tensor(sizes, dtype=torch.long, device=device)
            node_type = torch.arange(len(sizes), device=device)
            data.node_type = node_type.repeat_interleave(sizes)

        if add_edge_type and edge_index is not None:
            sizes = [offset[1] - offset[0] for offset in edge_slices.values()]
            sizes = torch.tensor(sizes, dtype=torch.long, device=device)
            edge_type = torch.arange(len(sizes), device=device)
            data.edge_type = edge_type.repeat_interleave(sizes)

        return data

    # FeatureStore interface ##################################################

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        if not attr.is_set('index'):
            attr.index = None

        out = self._node_store_dict.get(attr.group_name, None)
        if out:
            # Group name exists, handle index or create new attribute name:
            val = getattr(out, attr.attr_name, None)
            if val is not None:
                val[attr.index] = tensor
            else:
                assert attr.index is None
                setattr(self[attr.group_name], attr.attr_name, tensor)
        else:
            # No node storage found, just store tensor in new one:
            setattr(self[attr.group_name], attr.attr_name, tensor)
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        # Retrieve tensor and index accordingly:
        tensor = getattr(self[attr.group_name], attr.attr_name, None)
        if tensor is not None:
            # TODO this behavior is a bit odd, since TensorAttr requires that
            # we set `index`. So, we assume here that indexing by `None` is
            # equivalent to not indexing at all, which is not in line with
            # Python semantics.
            return tensor[attr.index] if attr.index is not None else tensor
        return None

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        # Remove tensor entirely:
        if hasattr(self[attr.group_name], attr.attr_name):
            delattr(self[attr.group_name], attr.attr_name)
            return True
        return False

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        out = []
        for group_name, group in self.node_items():
            for attr_name in group:
                if group.is_node_attr(attr_name):
                    out.append(TensorAttr(group_name, attr_name))
        return out

    # GraphStore interface ####################################################

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        if not hasattr(self, '_edge_attrs'):
            self._edge_attrs = {}
        self._edge_attrs[(edge_attr.edge_type, edge_attr.layout)] = edge_attr

        row, col = edge_index
        store = self[edge_attr.edge_type]

        if edge_attr.layout == EdgeLayout.COO:
            store.edge_index = torch.stack([row, col], dim=0)
        elif edge_attr.layout == EdgeLayout.CSR:
            store.adj = SparseTensor(
                rowptr=row,
                col=col,
                sparse_sizes=edge_attr.size,
                is_sorted=True,
                trust_data=True,
            )
        else:  # edge_attr.layout == EdgeLayout.CSC:
            size = edge_attr.size[::-1] if edge_attr.size is not None else None
            store.adj_t = SparseTensor(
                rowptr=col,
                col=row,
                sparse_sizes=size,
                is_sorted=True,
                trust_data=True,
            )
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        r"""Gets an edge index from edge storage, in the specified layout."""
        store = self[edge_attr.edge_type]

        edge_attrs = getattr(self, '_edge_attrs', {})
        if (edge_attr.edge_type, edge_attr.layout) in edge_attrs:
            edge_attr = edge_attrs[(edge_attr.edge_type, edge_attr.layout)]
        if edge_attr.size is None:
            edge_attr.size = store.size()  # Modify in-place.

        if edge_attr.layout == EdgeLayout.COO and 'edge_index' in store:
            row, col = store.edge_index
            return row, col
        elif edge_attr.layout == EdgeLayout.CSR and 'adj' in store:
            rowptr, col, _ = store.adj.csr()
            return rowptr, col
        elif edge_attr.layout == EdgeLayout.CSC and 'adj_t' in store:
            colptr, row, _ = store.adj_t.csr()
            return row, colptr
        return None

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        edge_type = edge_attr.edge_type
        store = self[edge_type]
        if edge_attr.layout == EdgeLayout.COO and 'edge_index' in store:
            del store.edge_index
            if hasattr(self, '_edge_attrs'):
                self._edge_attrs.pop((edge_type, EdgeLayout.COO), None)
            return True
        elif edge_attr.layout == EdgeLayout.CSR and 'adj' in store:
            del store.adj
            if hasattr(self, '_edge_attrs'):
                self._edge_attrs.pop((edge_type, EdgeLayout.CSR), None)
            return True
        elif edge_attr.layout == EdgeLayout.CSC and 'adj_t' in store:
            del store.adj_t
            if hasattr(self, '_edge_attrs'):
                self._edge_attrs.pop((edge_type, EdgeLayout.CSC), None)
            return True
        return False

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        edge_attrs = getattr(self, '_edge_attrs', {})

        for store in self.edge_stores:
            if ('edge_index' in store
                    and (store._key, EdgeLayout.COO) not in edge_attrs):
                edge_attrs[(store._key, EdgeLayout.COO)] = EdgeAttr(
                    store._key, 'coo', is_sorted=False)
            if ('adj' in store
                    and (store._key, EdgeLayout.CSR) not in edge_attrs):
                size = store.adj.sparse_sizes()
                edge_attrs[(store._key, EdgeLayout.CSR)] = EdgeAttr(
                    store._key, 'csr', size=size)
            if ('adj_t' in store
                    and (store._key, EdgeLayout.CSC) not in edge_attrs):
                size = store.adj_t.sparse_sizes()[::-1]
                edge_attrs[(store._key, EdgeLayout.CSC)] = EdgeAttr(
                    store._key, 'csc', size=size)

        return list(edge_attrs.values())


# Helper functions ############################################################


def get_node_slices(num_nodes: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    r"""Returns the boundaries of each node type in a graph."""
    node_slices: Dict[NodeType, Tuple[int, int]] = {}
    cumsum = 0
    for node_type, N in num_nodes.items():
        node_slices[node_type] = (cumsum, cumsum + N)
        cumsum += N
    return node_slices


def offset_edge_index(
    node_slices: Dict[NodeType, Tuple[int, int]],
    edge_type: EdgeType,
    edge_index: Tensor,
) -> Tensor:
    r"""Increases the edge indices by the offsets of source and destination
    node types.
    """
    src, _, dst = edge_type
    offset = [[node_slices[src][0]], [node_slices[dst][0]]]
    offset = torch.tensor(offset, device=edge_index.device)
    return edge_index + offset


def to_homogeneous_edge_index(
    data: HeteroData,
) -> Tuple[Optional[Tensor], Dict[NodeType, Any], Dict[EdgeType, Any]]:
    r"""Converts a heterogeneous graph into a homogeneous typed graph."""
    # Record slice information per node type:
    node_slices = get_node_slices(data.num_nodes_dict)

    # Record edge indices and slice information per edge type:
    cumsum = 0
    edge_indices: List[Tensor] = []
    edge_slices: Dict[EdgeType, Tuple[int, int]] = {}
    for edge_type, edge_index in data.collect('edge_index', True).items():
        edge_index = offset_edge_index(node_slices, edge_type, edge_index)
        edge_indices.append(edge_index)
        edge_slices[edge_type] = (cumsum, cumsum + edge_index.size(1))
        cumsum += edge_index.size(1)

    edge_index: Optional[Tensor] = None
    if len(edge_indices) == 1:  # Memory-efficient `torch.cat`:
        edge_index = edge_indices[0]
    elif len(edge_indices) > 1:
        edge_index = torch.cat(edge_indices, dim=-1)

    return edge_index, node_slices, edge_slices
