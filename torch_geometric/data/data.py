import copy
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import EdgeAttr, FeatureStore, GraphStore, TensorAttr
from torch_geometric.data.feature_store import _field_status
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.data.storage import (
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage,
)
from torch_geometric.deprecation import deprecated
from torch_geometric.typing import (
    EdgeTensorType,
    EdgeType,
    FeatureTensorType,
    NodeType,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils import is_sparse, select, subgraph


class BaseData:
    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def stores_as(self, data: 'BaseData'):
        raise NotImplementedError

    @property
    def stores(self) -> List[BaseStorage]:
        raise NotImplementedError

    @property
    def node_stores(self) -> List[NodeStorage]:
        raise NotImplementedError

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        raise NotImplementedError

    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        raise NotImplementedError

    def update(self, data: 'BaseData') -> 'BaseData':
        r"""Updates the data object with the elements from another data object.
        """
        raise NotImplementedError

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the dimension for which the value :obj:`value` of the
        attribute :obj:`key` will get concatenated when creating mini-batches
        using :class:`torch_geometric.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the incremental count to cumulatively increase the value
        :obj:`value` of the attribute :obj:`key` when creating mini-batches
        using :class:`torch_geometric.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def debug(self):
        raise NotImplementedError

    ###########################################################################

    @property
    def keys(self) -> List[str]:
        r"""Returns a list of all graph attribute names."""
        out = []
        for store in self.stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph.

        .. note::
            The number of nodes in the data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases, however, a graph may only be given without any
            node-level attributes.
            :pyg:`PyG` then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        try:
            return sum([v.num_nodes for v in self.node_stores])
        except TypeError:
            return None

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""Returns the size of the adjacency matrix induced by the graph."""
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

    @property
    def num_edges(self) -> int:
        r"""Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges."""
        return sum([v.num_edges for v in self.edge_stores])

    def node_attrs(self) -> List[str]:
        r"""Returns all node-level tensor attribute names."""
        return list(set(chain(*[s.node_attrs() for s in self.node_stores])))

    def edge_attrs(self) -> List[str]:
        r"""Returns all edge-level tensor attribute names."""
        return list(set(chain(*[s.edge_attrs() for s in self.edge_stores])))

    def is_coalesced(self) -> bool:
        r"""Returns :obj:`True` if edge indices :obj:`edge_index` are sorted
        and do not contain duplicate entries."""
        return all([store.is_coalesced() for store in self.edge_stores])

    def generate_ids(self):
        r"""Generates and sets :obj:`n_id` and :obj:`e_id` attributes to assign
        each node and edge to a continuously ascending and unique ID."""
        for store in self.node_stores:
            store.n_id = torch.arange(store.num_nodes)
        for store in self.edge_stores:
            store.e_id = torch.arange(store.num_edges)

    def coalesce(self):
        r"""Sorts and removes duplicated entries from edge indices
        :obj:`edge_index`."""
        for store in self.edge_stores:
            store.coalesce()
        return self

    def has_isolated_nodes(self) -> bool:
        r"""Returns :obj:`True` if the graph contains isolated nodes."""
        return any([store.has_isolated_nodes() for store in self.edge_stores])

    def has_self_loops(self) -> bool:
        """Returns :obj:`True` if the graph contains self-loops."""
        return any([store.has_self_loops() for store in self.edge_stores])

    def is_undirected(self) -> bool:
        r"""Returns :obj:`True` if graph edges are undirected."""
        return all([store.is_undirected() for store in self.edge_stores])

    def is_directed(self) -> bool:
        r"""Returns :obj:`True` if graph edges are directed."""
        return not self.is_undirected()

    def apply_(self, func: Callable, *args: List[str]):
        r"""Applies the in-place function :obj:`func`, either to all attributes
        or only the ones given in :obj:`*args`."""
        for store in self.stores:
            store.apply_(func, *args)
        return self

    def apply(self, func: Callable, *args: List[str]):
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`."""
        for store in self.stores:
            store.apply(func, *args)
        return self

    def clone(self, *args: List[str]):
        r"""Performs cloning of tensors, either for all attributes or only the
        ones given in :obj:`*args`."""
        return copy.copy(self).apply(lambda x: x.clone(), *args)

    def contiguous(self, *args: List[str]):
        r"""Ensures a contiguous memory layout, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.contiguous(), *args)

    def to(self, device: Union[int, str], *args: List[str],
           non_blocking: bool = False):
        r"""Performs tensor device conversion, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: List[str]):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(self, device: Optional[Union[int, str]] = None, *args: List[str],
             non_blocking: bool = False):
        r"""Copies attributes to CUDA memory, either for all attributes or only
        the ones given in :obj:`*args`."""
        # Some PyTorch tensor like objects require a default value for `cuda`:
        device = 'cuda' if device is None else device
        return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking),
                          *args)

    def pin_memory(self, *args: List[str]):
        r"""Copies attributes to pinned memory, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: List[str]):
        r"""Moves attributes to shared memory, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply_(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: List[str]):
        r"""Detaches attributes from the computation graph, either for all
        attributes or only the ones given in :obj:`*args`."""
        return self.apply_(lambda x: x.detach_(), *args)

    def detach(self, *args: List[str]):
        r"""Detaches attributes from the computation graph by creating a new
        tensor, either for all attributes or only the ones given in
        :obj:`*args`."""
        return self.apply(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: List[str], requires_grad: bool = True):
        r"""Tracks gradient computation, either for all attributes or only the
        ones given in :obj:`*args`."""
        return self.apply_(
            lambda x: x.requires_grad_(requires_grad=requires_grad), *args)

    def record_stream(self, stream: torch.cuda.Stream, *args: List[str]):
        r"""Ensures that the tensor memory is not reused for another tensor
        until all current work queued on :obj:`stream` has been completed,
        either for all attributes or only the ones given in :obj:`*args`."""
        return self.apply_(lambda x: x.record_stream(stream), *args)

    @property
    def is_cuda(self) -> bool:
        r"""Returns :obj:`True` if any :class:`torch.Tensor` attribute is
        stored on the GPU, :obj:`False` otherwise."""
        for store in self.stores:
            for value in store.values():
                if isinstance(value, Tensor) and value.is_cuda:
                    return True
        return False

    # Deprecated functions ####################################################

    @deprecated(details="use 'has_isolated_nodes' instead")
    def contains_isolated_nodes(self) -> bool:
        return self.has_isolated_nodes()

    @deprecated(details="use 'has_self_loops' instead")
    def contains_self_loops(self) -> bool:
        return self.has_self_loops()


###############################################################################


@dataclass
class DataTensorAttr(TensorAttr):
    r"""Tensor attribute for `Data` without group name."""
    def __init__(
        self,
        attr_name=_field_status.UNSET,
        index=None,
    ):
        super().__init__(None, attr_name, index)


@dataclass
class DataEdgeAttr(EdgeAttr):
    r"""Edge attribute class for `Data` without edge type."""
    def __init__(
        self,
        layout: Optional[EdgeLayout] = None,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(None, layout, is_sorted, size)


###############################################################################


class Data(BaseData, FeatureStore, GraphStore):
    r"""A data object describing a homogeneous graph.
    The data object can hold node-level, link-level and graph-level attributes.
    In general, :class:`~torch_geometric.data.Data` tries to mimic the
    behavior of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/get_started/
    introduction.html#data-handling-of-graphs>`__ for the accompanying
    tutorial.

    .. code-block:: python

        from torch_geometric.data import Data

        data = Data(x=x, edge_index=edge_index, ...)

        # Add additional arguments to `data`:
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)

        # Analyzing the graph structure:
        data.num_nodes
        >>> 23

        data.is_directed()
        >>> False

        # PyTorch tensor functionality:
        data = data.pin_memory()
        data = data.to('cuda:0', non_blocking=True)

    Args:
        x (torch.Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (torch.Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (torch.Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        pos (torch.Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        # `Data` doesn't support group_name, so we need to adjust `TensorAttr`
        # accordingly here to avoid requiring `group_name` to be set:
        super().__init__(tensor_attr_cls=DataTensorAttr)

        # `Data` doesn't support edge_type, so we need to adjust `EdgeAttr`
        # accordingly here to avoid requiring `edge_type` to be set:
        GraphStore.__init__(self, edge_attr_cls=DataEdgeAttr)

        self.__dict__['_store'] = GlobalStorage(_parent=self)

        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if pos is not None:
            self.pos = pos

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, 'fset', None) is not None:
            propobj.fset(self, value)
        else:
            setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    # TODO consider supporting the feature store interface for
    # __getitem__, __setitem__, and __delitem__ so, for example, we
    # can accept key: Union[str, TensorAttr] in __getitem__.
    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        has_dict = any([isinstance(v, Mapping) for v in self._store.values()])

        if not has_dict:
            info = [size_repr(k, v) for k, v in self._store.items()]
            info = ', '.join(info)
            return f'{cls}({info})'
        else:
            info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
            info = ',\n'.join(info)
            return f'{cls}(\n{info}\n)'

    def stores_as(self, data: 'Data'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    def update(self, data: Union['Data', Dict[str, Any]]) -> 'Data':
        for key, value in data.items():
            self[key] = value
        return self

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if is_sparse(value) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        num_nodes = self.num_nodes
        if num_nodes is None:
            status = False
            warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'",
                          raise_on_error)

        if 'edge_index' in self:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                status = False
                warn_or_raise(
                    f"'edge_index' needs to be of shape [2, num_edges] in "
                    f"'{cls_name}' (found {self.edge_index.size()})",
                    raise_on_error)

        if 'edge_index' in self and self.edge_index.numel() > 0:
            if self.edge_index.min() < 0:
                status = False
                warn_or_raise(
                    f"'edge_index' contains negative indices in "
                    f"'{cls_name}' (found {int(self.edge_index.min())})",
                    raise_on_error)

            if num_nodes is not None and self.edge_index.max() >= num_nodes:
                status = False
                warn_or_raise(
                    f"'edge_index' contains larger indices than the number "
                    f"of nodes ({num_nodes}) in '{cls_name}' "
                    f"(found {int(self.edge_index.max())})", raise_on_error)

        return status

    def debug(self):
        pass  # TODO

    def is_node_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes a
        node-level tensor attribute."""
        return self._store.is_node_attr(key)

    def is_edge_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes an
        edge-level tensor attribute."""
        return self._store.is_edge_attr(key)

    def subgraph(self, subset: Tensor) -> 'Data':
        r"""Returns the induced subgraph given by the node indices
        :obj:`subset`.

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        out = subgraph(subset, self.edge_index, relabel_nodes=True,
                       num_nodes=self.num_nodes, return_edge_mask=True)
        edge_index, _, edge_mask = out

        data = copy.copy(self)

        for key, value in self:
            if key == 'edge_index':
                data.edge_index = edge_index
            elif key == 'num_nodes':
                if subset.dtype == torch.bool:
                    data.num_nodes = int(subset.sum())
                else:
                    data.num_nodes = subset.size(0)
            elif self.is_node_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, subset, dim=cat_dim)
            elif self.is_edge_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, edge_mask, dim=cat_dim)

        return data

    def edge_subgraph(self, subset: Tensor) -> 'Data':
        r"""Returns the induced subgraph given by the edge indices
        :obj:`subset`.
        Will currently preserve all the nodes in the graph, even if they are
        isolated after subgraph computation.

        Args:
            subset (LongTensor or BoolTensor): The edges to keep.
        """
        data = copy.copy(self)

        for key, value in self:
            if self.is_edge_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, subset, dim=cat_dim)

        return data

    def to_heterogeneous(
        self,
        node_type: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_type_names: Optional[List[NodeType]] = None,
        edge_type_names: Optional[List[EdgeType]] = None,
    ):
        r"""Converts a :class:`~torch_geometric.data.Data` object to a
        heterogeneous :class:`~torch_geometric.data.HeteroData` object.
        For this, node and edge attributes are splitted according to the
        node-level and edge-level vectors :obj:`node_type` and
        :obj:`edge_type`, respectively.
        :obj:`node_type_names` and :obj:`edge_type_names` can be used to give
        meaningful node and edge type names, respectively.
        That is, the node_type :obj:`0` is given by :obj:`node_type_names[0]`.
        If the :class:`~torch_geometric.data.Data` object was constructed via
        :meth:`~torch_geometric.data.HeteroData.to_homogeneous`, the object can
        be reconstructed without any need to pass in additional arguments.

        Args:
            node_type (torch.Tensor, optional): A node-level vector denoting
                the type of each node. (default: :obj:`None`)
            edge_type (torch.Tensor, optional): An edge-level vector denoting
                the type of each edge. (default: :obj:`None`)
            node_type_names (List[str], optional): The names of node types.
                (default: :obj:`None`)
            edge_type_names (List[Tuple[str, str, str]], optional): The names
                of edge types. (default: :obj:`None`)
        """
        from torch_geometric.data import HeteroData

        if node_type is None:
            node_type = self._store.get('node_type', None)
        if node_type is None:
            node_type = torch.zeros(self.num_nodes, dtype=torch.long)

        if node_type_names is None:
            store = self._store
            node_type_names = store.__dict__.get('_node_type_names', None)
        if node_type_names is None:
            node_type_names = [str(i) for i in node_type.unique().tolist()]

        if edge_type is None:
            edge_type = self._store.get('edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(self.num_edges, dtype=torch.long)

        if edge_type_names is None:
            store = self._store
            edge_type_names = store.__dict__.get('_edge_type_names', None)
        if edge_type_names is None:
            edge_type_names = []
            edge_index = self.edge_index
            for i in edge_type.unique().tolist():
                src, dst = edge_index[:, edge_type == i]
                src_types = node_type[src].unique().tolist()
                dst_types = node_type[dst].unique().tolist()
                if len(src_types) != 1 and len(dst_types) != 1:
                    raise ValueError(
                        "Could not construct a 'HeteroData' object from the "
                        "'Data' object because single edge types span over "
                        "multiple node types")
                edge_type_names.append((node_type_names[src_types[0]], str(i),
                                        node_type_names[dst_types[0]]))

        # We iterate over node types to find the local node indices belonging
        # to each node type. Furthermore, we create a global `index_map` vector
        # that maps global node indices to local ones in the final
        # heterogeneous graph:
        node_ids, index_map = {}, torch.empty_like(node_type)
        for i, key in enumerate(node_type_names):
            node_ids[i] = (node_type == i).nonzero(as_tuple=False).view(-1)
            index_map[node_ids[i]] = torch.arange(len(node_ids[i]),
                                                  device=index_map.device)

        # We iterate over edge types to find the local edge indices:
        edge_ids = {}
        for i, key in enumerate(edge_type_names):
            edge_ids[i] = (edge_type == i).nonzero(as_tuple=False).view(-1)

        data = HeteroData()

        for i, key in enumerate(node_type_names):
            for attr, value in self.items():
                if attr in {'node_type', 'edge_type', 'ptr'}:
                    continue
                elif isinstance(value, Tensor) and self.is_node_attr(attr):
                    cat_dim = self.__cat_dim__(attr, value)
                    data[key][attr] = value.index_select(cat_dim, node_ids[i])

            if len(data[key]) == 0:
                data[key].num_nodes = node_ids[i].size(0)

        for i, key in enumerate(edge_type_names):
            src, _, dst = key
            for attr, value in self.items():
                if attr in {'node_type', 'edge_type', 'ptr'}:
                    continue
                elif attr == 'edge_index':
                    edge_index = value[:, edge_ids[i]]
                    edge_index[0] = index_map[edge_index[0]]
                    edge_index[1] = index_map[edge_index[1]]
                    data[key].edge_index = edge_index
                elif isinstance(value, Tensor) and self.is_edge_attr(attr):
                    cat_dim = self.__cat_dim__(attr, value)
                    data[key][attr] = value.index_select(cat_dim, edge_ids[i])

        # Add global attributes.
        exclude_keys = set(data.keys) | {
            'node_type', 'edge_type', 'edge_index', 'num_nodes', 'ptr'
        }
        for attr, value in self.items():
            if attr in exclude_keys:
                continue
            data[attr] = value

        return data

    ###########################################################################

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> 'Data':
        r"""Creates a :class:`~torch_geometric.data.Data` object from a Python
        dictionary."""
        return cls(**mapping)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the graph."""
        return self._store.num_node_features

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the graph.
        Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the graph."""
        return self._store.num_edge_features

    @property
    def num_node_types(self) -> int:
        r"""Returns the number of node types in the graph."""
        return int(self.node_type.max()) + 1 if 'node_type' in self else 1

    @property
    def num_edge_types(self) -> int:
        r"""Returns the number of edge types in the graph."""
        return int(self.edge_type.max()) + 1 if 'edge_type' in self else 1

    def __iter__(self) -> Iterable:
        r"""Iterates over all attributes in the data, yielding their attribute
        names and values."""
        for key, value in self._store.items():
            yield key, value

    def __call__(self, *args: List[str]) -> Iterable:
        r"""Iterates over all attributes :obj:`*args` in the data, yielding
        their attribute names and values.
        If :obj:`*args` is not given, will iterate over all attributes."""
        for key, value in self._store.items(*args):
            yield key, value

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def edge_attr(self) -> Any:
        return self['edge_attr'] if 'edge_attr' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    # Deprecated functions ####################################################

    @property
    @deprecated(details="use 'data.face.size(-1)' instead")
    def num_faces(self) -> Optional[int]:
        r"""Returns the number of faces in the mesh."""
        if 'face' in self._store and isinstance(self.face, Tensor):
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    # FeatureStore interface ##################################################

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        out = self.get(attr.attr_name)
        if out is not None and attr.index is not None:
            out[attr.index] = tensor
        else:
            assert attr.index is None
            setattr(self, attr.attr_name, tensor)
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        tensor = getattr(self, attr.attr_name, None)
        if tensor is not None:
            # TODO this behavior is a bit odd, since TensorAttr requires that
            # we set `index`. So, we assume here that indexing by `None` is
            # equivalent to not indexing at all, which is not in line with
            # Python semantics.
            return tensor[attr.index] if attr.index is not None else tensor
        return None

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        if hasattr(self, attr.attr_name):
            delattr(self, attr.attr_name)
            return True
        return False

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        r"""Obtains all feature attributes stored in `Data`."""
        return [
            TensorAttr(attr_name=name) for name in self._store.keys()
            if self._store.is_node_attr(name)
        ]

    # GraphStore interface ####################################################

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        if not hasattr(self, '_edge_attrs'):
            self._edge_attrs = {}
        self._edge_attrs[edge_attr.layout] = edge_attr

        row, col = edge_index

        if edge_attr.layout == EdgeLayout.COO:
            self.edge_index = torch.stack([row, col], dim=0)
        elif edge_attr.layout == EdgeLayout.CSR:
            self.adj = SparseTensor(
                rowptr=row,
                col=col,
                sparse_sizes=edge_attr.size,
                is_sorted=True,
                trust_data=True,
            )
        else:  # edge_attr.layout == EdgeLayout.CSC:
            size = edge_attr.size[::-1] if edge_attr.size is not None else None
            self.adj_t = SparseTensor(
                rowptr=col,
                col=row,
                sparse_sizes=size,
                is_sorted=True,
                trust_data=True,
            )
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        if edge_attr.size is None:
            edge_attr.size = self.size()  # Modify in-place.

        if edge_attr.layout == EdgeLayout.COO and 'edge_index' in self:
            row, col = self.edge_index
            return row, col
        elif edge_attr.layout == EdgeLayout.CSR and 'adj' in self:
            rowptr, col, _ = self.adj.csr()
            return rowptr, col
        elif edge_attr.layout == EdgeLayout.CSC and 'adj_t' in self:
            colptr, row, _ = self.adj_t.csr()
            return row, colptr
        return None

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        if edge_attr.layout == EdgeLayout.COO and 'edge_index' in self:
            del self.edge_index
            if hasattr(self, '_edge_attrs'):
                self._edges_to_layout.pop(EdgeLayout.COO, None)
            return True
        elif edge_attr.layout == EdgeLayout.CSR and 'adj' in self:
            del self.adj
            if hasattr(self, '_edge_attrs'):
                self._edges_to_layout.pop(EdgeLayout.CSR, None)
            return True
        elif edge_attr.layout == EdgeLayout.CSC and 'adj_t' in self:
            del self.adj_t
            if hasattr(self, '_edge_attrs'):
                self._edges_to_layout.pop(EdgeLayout.CSC, None)
            return True
        return False

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        edge_attrs = getattr(self, '_edge_attrs', {})

        if 'edge_index' in self and EdgeLayout.COO not in edge_attrs:
            edge_attrs[EdgeLayout.COO] = DataEdgeAttr('coo', is_sorted=False)
        if 'adj' in self and EdgeLayout.CSR not in edge_attrs:
            size = self.adj.sparse_sizes()
            edge_attrs[EdgeLayout.CSR] = DataEdgeAttr('csr', size=size)
        if 'adj_t' in self and EdgeLayout.CSC not in edge_attrs:
            size = self.adj_t.sparse_sizes()[::-1]
            edge_attrs[EdgeLayout.CSC] = DataEdgeAttr('csc', size=size)

        return list(edge_attrs.values())


###############################################################################


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = ' ' * indent
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, SparseTensor):
        out = str(value.sizes())[:-1] + f', nnz={value.nnz()}]'
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = '{}'
    elif (isinstance(value, Mapping) and len(value) == 1
          and not isinstance(list(value.values())[0], Mapping)):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, BaseStorage):
        return f'{pad}\033[1m{key}\033[0m={out}'
    else:
        return f'{pad}{key}={out}'


def warn_or_raise(msg: str, raise_on_error: bool = True):
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg)
