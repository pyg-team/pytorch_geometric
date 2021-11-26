from typing import (Optional, Dict, Any, Union, List, Iterable, Tuple,
                    NamedTuple, Callable)
from torch_geometric.typing import OptTensor, NodeType, EdgeType
from torch_geometric.deprecation import deprecated

import copy
from collections.abc import Sequence, Mapping

import torch
import numpy as np
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.utils import subgraph
from torch_geometric.data.storage import (BaseStorage, NodeStorage,
                                          EdgeStorage, GlobalStorage)


class BaseData(object):
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
            PyG then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behaviour.
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

    def is_coalesced(self) -> bool:
        r"""Returns :obj:`True` if edge indices :obj:`edge_index` are sorted
        and do not contain duplicate entries."""
        return all([store.is_coalesced() for store in self.edge_stores])

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

    def clone(self):
        r"""Performs a deep-copy of the data object."""
        return copy.deepcopy(self)

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

    # Deprecated functions ####################################################

    @deprecated(details="use 'has_isolated_nodes' instead")
    def contains_isolated_nodes(self) -> bool:
        return self.has_isolated_nodes()

    @deprecated(details="use 'has_self_loops' instead")
    def contains_self_loops(self) -> bool:
        return self.has_self_loops()


###############################################################################


class Data(BaseData):
    r"""A data object describing a homogeneous graph.
    The data object can hold node-level, link-level and graph-level attributes.
    In general, :class:`~torch_geometric.data.Data` tries to mimic the
    behaviour of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
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
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__()
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
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

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

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def debug(self):
        pass  # TODO

    def is_node_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes a
        node-level attribute."""
        value = self[key]
        cat_dim = self.__cat_dim__(key, value, self._store)

        num_nodes, num_edges = self.num_nodes, self.num_edges
        if not isinstance(value, torch.Tensor):
            return False
        if value.size(cat_dim) != num_nodes:
            return False
        if num_nodes != num_edges:
            return True
        return 'edge' not in key

    def is_edge_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes an
        edge-level attribute."""
        value = self[key]
        cat_dim = self.__cat_dim__(key, value, self._store)

        num_nodes, num_edges = self.num_nodes, self.num_edges
        if not isinstance(value, torch.Tensor):
            return False
        if value.size(cat_dim) != num_edges:
            return False
        if num_nodes != num_edges:
            return True
        return 'edge' in key

    def subgraph(self, subset: Tensor):
        r"""Returns the induced subgraph given by the node indices
        :obj:`subset`.

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """

        out = subgraph(subset, self.edge_index, relabel_nodes=True,
                       num_nodes=self.num_nodes, return_edge_mask=True)
        edge_index, _, edge_mask = out

        if subset.dtype == torch.bool:
            num_nodes = int(subset.sum())
        else:
            num_nodes = subset.size(0)

        data = copy.copy(self)

        for key, value in data:
            if key == 'edge_index':
                data.edge_index = edge_index
            elif key == 'num_nodes':
                data.num_nodes = num_nodes
            elif isinstance(value, Tensor):
                if self.is_node_attr(key):
                    data[key] = value[subset]
                elif self.is_edge_attr(key):
                    data[key] = value[edge_mask]

        return data

    def to_heterogeneous(self, node_type: Optional[Tensor] = None,
                         edge_type: Optional[Tensor] = None,
                         node_type_names: Optional[List[NodeType]] = None,
                         edge_type_names: Optional[List[EdgeType]] = None):
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
            node_type (Tensor, optional): A node-level vector denoting the type
                of each node. (default: :obj:`None`)
            edge_type (Tensor, optional): An edge-level vector denoting the
                type of each edge. (default: :obj:`None`)
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
            index_map[node_ids[i]] = torch.arange(len(node_ids[i]))

        # We iterate over edge types to find the local edge indices:
        edge_ids = {}
        for i, key in enumerate(edge_type_names):
            edge_ids[i] = (edge_type == i).nonzero(as_tuple=False).view(-1)

        data = HeteroData()

        for i, key in enumerate(node_type_names):
            for attr, value in self.items():
                if attr == 'node_type' or attr == 'edge_type':
                    continue
                elif isinstance(value, Tensor) and self.is_node_attr(attr):
                    data[key][attr] = value[node_ids[i]]

            if len(data[key]) == 0:
                data[key].num_nodes = node_ids[i].size(0)

        for i, key in enumerate(edge_type_names):
            src, _, dst = key
            for attr, value in self.items():
                if attr == 'node_type' or attr == 'edge_type':
                    continue
                elif attr == 'edge_index':
                    edge_index = value[:, edge_ids[i]]
                    edge_index[0] = index_map[edge_index[0]]
                    edge_index[1] = index_map[edge_index[1]]
                    data[key].edge_index = edge_index
                elif isinstance(value, Tensor) and self.is_edge_attr(attr):
                    data[key][attr] = value[edge_ids[i]]

        # Add global attributes.
        keys = set(data.keys) | {'node_type', 'edge_type', 'num_nodes'}
        for attr, value in self.items():
            if attr in keys:
                continue
            if len(data.node_stores) == 1:
                data.node_stores[0][attr] = value
            else:
                data[attr] = value

        return data

    ###########################################################################

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]):
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
        return self._store.num_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the graph."""
        return self._store.num_edge_features

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
        if 'face' in self._store and isinstance(self.face, torch.Tensor):
            return self.face.size(self.__cat_dim__('face', self.face))
        return None


###############################################################################


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = ' ' * indent
    if isinstance(value, torch.Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, torch.Tensor):
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
