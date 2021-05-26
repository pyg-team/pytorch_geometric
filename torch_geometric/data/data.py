from typing import (Optional, Dict, Any, Union, List, Iterable, NamedTuple,
                    Callable)
from torch_geometric.typing import OptTensor
from torch_geometric.deprecation import deprecated

import copy
from collections.abc import Sequence, Mapping

import torch
import numpy as np
from torch_sparse import SparseTensor

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

    @property
    def _stores(self) -> List[BaseStorage]:
        raise NotImplementedError

    @property
    def _node_stores(self) -> List[NodeStorage]:
        raise NotImplementedError

    @property
    def _edge_stores(self) -> List[EdgeStorage]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_namedtuple(self) -> NamedTuple:
        raise NotImplementedError

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def debug(self):
        raise NotImplementedError

    ###########################################################################

    @property
    def keys(self) -> List[str]:
        r"""Returns a list of graph attribute names."""
        out = []
        for store in self._stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
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
            The number of nodes in your data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number does not have to be correct
            which can result in unexpected issues in return.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        try:
            return sum([v.num_nodes for v in self._node_stores])
        except TypeError:
            return None

    @property
    def num_edges(self) -> int:
        r"""Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges."""
        return sum([v.num_edges for v in self._edge_stores])

    def is_coalesced(self) -> bool:
        r"""Returns :obj:`True`, if edge indices :obj:`edge_index` are sorted
        and do not contain duplicate entries."""
        return all([store.is_coalesced() for store in self._edge_stores])

    def coalesce(self):
        r""""Sorts and removes duplicated entries from edge indices
        :obj:`edge_index`."""
        for store in self._edge_stores:
            store.coalesce()
        return self

    def has_isolated_nodes(self) -> bool:
        r"""Returns :obj:`True`, if the graph contains isolated nodes."""
        return any([store.has_isolated_nodes() for store in self._edge_stores])

    def has_self_loops(self) -> bool:
        """Returns :obj:`True`, if the graph contains self-loops."""
        return any([store.has_self_loops() for store in self._edge_stores])

    def is_undirected(self) -> bool:
        r"""Returns :obj:`True`, if graph edges are undirected."""
        return all([store.is_undirected() for store in self._edge_stores])

    def is_directed(self) -> bool:
        r"""Returns :obj:`True`, if graph edges are directed."""
        return not self.is_undirected()

    def clone(self):
        r"""Performs a deep-copy of the data object."""
        return copy.deepcopy(self)

    def apply(self, func: Callable, *args: List[str]):
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`."""
        for store in self._stores:
            store.apply(func, *args)
        return self

    def contiguous(self, *args: List[str]):
        r"""Ensures a contiguous memory layout, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.contiguous(), *args)

    def to(self, device: Union[int, str], *args: List[str],
           non_blocking: bool = False):
        r"""Performs tensor dtype and/or device conversion, either for all
        attributes or only the ones given in :obj:`*args`."""
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: List[str]):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(self, device: Union[int, str], *args: List[str],
             non_blocking: bool = False):
        r"""Copies attributes to CUDA memory, either for all attributes or only
        the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.cuda(non_blocking=non_blocking), *args)

    def pin_memory(self, *args: List[str]):
        r"""Copies attributes to pinned memory, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: List[str]):
        r"""Moves attributes to shared memory, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: List[str]):
        r"""Detaches attributes from the computation graph, either for all
        attributes or only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.detach_(), *args)

    def detach(self, *args: List[str]):
        r"""Detaches attributes from the computation graph by creating a new
        tensor, either for all attributes or only the ones given in
        :obj:`*args`."""
        return self.apply(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: List[str], requires_grad: bool = True):
        r"""Tracks gradient computation, either for all attributes or only the
        ones given in :obj:`*args`."""
        return self.apply(
            lambda x: x.requires_grad_(requires_grad=requires_grad), *args)

    # Deprecated functions ####################################################

    @deprecated(details="use 'has_isolated_nodes' instead")
    def contains_isolated_nodes(self) -> bool:
        return self.has_isolated_nodes()

    @deprecated(details="use 'has_self_loops' instead")
    def contains_self_loops(self) -> bool:
        return self.has_self_loops()


###############################################################################


class Data(BaseData):
    r"""A plain old Python object modeling a single graph with various
    (optional) attributes:

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

    The data object is not restricted to these attributes and can be extented
    by any other additional data.

    Example::

        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__()
        self._store = GlobalStorage(_parent=self)
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        # `data._* = ...` => Link to the private `__dict__` store.
        # `data.* = ...` => Link to the `_store`.
        if key[:1] == '_':
            self.__dict__[key] = value
        else:
            setattr(self._store, key, value)

    def __delattr__(self, key: str):
        # `del data._*` => Link to the private `__dict__` store.
        # `del data.*` => Link to the `_store`.
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            delattr(self._store, key)

    def __getitem__(self, key: str) -> Any:
        r"""Gets the data of the attribute :obj:`key`."""
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        r"""Sets the attribute :obj:`key` to :obj:`value`."""
        self._store[key] = value

    def __delitem__(self, key: str):
        del self.store[key]

    def __copy__(self):
        out = self.__class__()
        for key, value in self.__dict__.items():
            if key not in ['_store']:
                out.__dict__[key] = value
        out._store = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        info = [size_repr(key, value) for key, value in self._store.items()]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))

    @property
    def _stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def _node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def _edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        return self._store.to_namedtuple()

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating mini-batches.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating mini-batches.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def debug(self):
        pass  # TODO

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
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
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
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def adj_t(self) -> Any:
        return self['adj_t'] if 'adj_t' in self._store else None

    @property
    def edge_attr(self) -> Any:
        return self['edge_attr'] if 'edge_attr' in self._store else None

    # Deprecated functions ####################################################

    @property
    @deprecated(details="use 'data.face.size(-1)' instead")
    def num_faces(self) -> Optional[int]:
        r"""Returns the number of faces in the mesh."""
        if 'face' in self._store:
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
    elif isinstance(value, Mapping) and len(value) == 1:
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [pad + size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, (NodeStorage, EdgeStorage)):
        return f'{pad}\033[1m{key}\033[0m={out}'
    else:
        return f'{pad}{key}={out}'
