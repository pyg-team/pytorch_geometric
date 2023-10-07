import copy
import warnings
import weakref
from collections import defaultdict, namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data.view import ItemsView, KeysView, ValuesView
from torch_geometric.typing import (
    EdgeType,
    NodeType,
    SparseTensor,
    TensorFrame,
)
from torch_geometric.utils import (
    coalesce,
    contains_isolated_nodes,
    is_torch_sparse_tensor,
    is_undirected,
    sort_edge_index,
)

N_KEYS = {'x', 'feat', 'pos', 'batch', 'node_type', 'n_id', 'tf'}
E_KEYS = {'edge_index', 'edge_weight', 'edge_attr', 'edge_type', 'e_id'}


class AttrType(Enum):
    NODE = 'NODE'
    EDGE = 'EDGE'
    OTHER = 'OTHER'


class BaseStorage(MutableMapping):
    # This class wraps a Python dictionary and extends it as follows:
    # 1. It allows attribute assignments, e.g.:
    #    `storage.x = ...` in addition to `storage['x'] = ...`
    # 2. It allows private attributes that are not exposed to the user, e.g.:
    #    `storage._{key} = ...` and accessible via `storage._{key}`
    # 3. It holds an (optional) weak reference to its parent object, e.g.:
    #    `storage._parent = weakref.ref(parent)`
    # 4. It allows iterating over only a subset of keys, e.g.:
    #    `storage.values('x', 'y')` or `storage.items('x', 'y')
    # 5. It adds additional PyTorch Tensor functionality, e.g.:
    #    `storage.cpu()`, `storage.cuda()` or `storage.share_memory_()`.
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        self._mapping = {}
        for key, value in (_mapping or {}).items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _key(self) -> Any:
        return None

    def _pop_cache(self, key: str):
        for cache in getattr(self, '_cached_attr', {}).values():
            cache.discard(key)

    def __len__(self) -> int:
        return len(self._mapping)

    def __getattr__(self, key: str) -> Any:
        if key == '_mapping':
            self._mapping = {}
            return self._mapping
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, 'fset', None) is not None:
            propobj.fset(self, value)
        elif key == '_parent':
            self.__dict__[key] = weakref.ref(value)
        elif key[:1] == '_':
            self.__dict__[key] = value
        else:
            self._pop_cache(key)
            self[key] = value

    def __delattr__(self, key: str):
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            self._pop_cache(key)
            del self[key]

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __setitem__(self, key: str, value: Any):
        self._pop_cache(key)
        if value is None and key in self._mapping:
            del self._mapping[key]
        elif value is not None:
            self._mapping[key] = value

    def __delitem__(self, key: str):
        if key in self._mapping:
            self._pop_cache(key)
            del self._mapping[key]

    def __iter__(self) -> Iterable:
        return iter(self._mapping)

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    def __getstate__(self) -> Dict[str, Any]:
        out = self.__dict__.copy()

        _parent = out.get('_parent', None)
        if _parent is not None:
            out['_parent'] = _parent()

        return out

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

        _parent = self.__dict__.get('_parent', None)
        if _parent is not None:
            self.__dict__['_parent'] = weakref.ref(_parent)

    def __repr__(self) -> str:
        return repr(self._mapping)

    # Allow iterating over subsets ############################################

    # In contrast to standard `keys()`, `values()` and `items()` functions of
    # Python dictionaries, we allow to only iterate over a subset of items
    # denoted by a list of keys `args`.
    # This is especially useful for adding PyTorch Tensor functionality to the
    # storage object, e.g., in case we only want to transfer a subset of keys
    # to the GPU (i.e. the ones that are relevant to the deep learning model).

    def keys(self, *args: str) -> KeysView:
        return KeysView(self._mapping, *args)

    def values(self, *args: str) -> ValuesView:
        return ValuesView(self._mapping, *args)

    def items(self, *args: str) -> ItemsView:
        return ItemsView(self._mapping, *args)

    def apply_(self, func: Callable, *args: str):
        r"""Applies the in-place function :obj:`func`, either to all attributes
        or only the ones given in :obj:`*args`."""
        for value in self.values(*args):
            recursive_apply_(value, func)
        return self

    def apply(self, func: Callable, *args: str):
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`."""
        for key, value in self.items(*args):
            self[key] = recursive_apply(value, func)
        return self

    # Additional functionality ################################################

    def get(self, key: str, value: Optional[Any] = None) -> Any:
        return self._mapping.get(key, value)

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        out_dict = copy.copy(self._mapping)
        # Needed to preserve individual `num_nodes` attributes when calling
        # `BaseData.collate`.
        # TODO (matthias) Try to make this more generic.
        if '_num_nodes' in self.__dict__:
            out_dict['_num_nodes'] = self.__dict__['_num_nodes']
        return out_dict

    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        field_names = list(self.keys())
        typename = f'{self.__class__.__name__}Tuple'
        StorageTuple = namedtuple(typename, field_names)
        return StorageTuple(*[self[key] for key in field_names])

    def clone(self, *args: str):
        r"""Performs a deep-copy of the object."""
        return copy.deepcopy(self)

    def contiguous(self, *args: str):
        r"""Ensures a contiguous memory layout, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.contiguous(), *args)

    def to(self, device: Union[int, str], *args: str,
           non_blocking: bool = False):
        r"""Performs tensor dtype and/or device conversion, either for all
        attributes or only the ones given in :obj:`*args`."""
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: str):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(self, device: Optional[Union[int, str]] = None, *args: str,
             non_blocking: bool = False):  # pragma: no cover
        r"""Copies attributes to CUDA memory, either for all attributes or only
        the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking),
                          *args)

    def pin_memory(self, *args: str):  # pragma: no cover
        r"""Copies attributes to pinned memory, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: str):
        r"""Moves attributes to shared memory, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: str):
        r"""Detaches attributes from the computation graph, either for all
        attributes or only the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.detach_(), *args)

    def detach(self, *args: str):
        r"""Detaches attributes from the computation graph by creating a new
        tensor, either for all attributes or only the ones given in
        :obj:`*args`."""
        return self.apply(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: str, requires_grad: bool = True):
        r"""Tracks gradient computation, either for all attributes or only the
        ones given in :obj:`*args`."""
        return self.apply(
            lambda x: x.requires_grad_(requires_grad=requires_grad), *args)

    def record_stream(self, stream: torch.cuda.Stream, *args: str):
        r"""Ensures that the tensor memory is not reused for another tensor
        until all current work queued on :obj:`stream` has been completed,
        either for all attributes or only the ones given in :obj:`*args`."""
        return self.apply_(lambda x: x.record_stream(stream), *args)


class NodeStorage(BaseStorage):
    @property
    def _key(self) -> NodeType:
        key = self.__dict__.get('_key', None)
        if key is None or not isinstance(key, str):
            raise ValueError("'_key' does not denote a valid node type")
        return key

    @property
    def can_infer_num_nodes(self):
        keys = set(self.keys())
        num_node_keys = {
            'num_nodes', 'x', 'pos', 'batch', 'adj', 'adj_t', 'edge_index',
            'face'
        }
        if len(keys & num_node_keys) > 0:
            return True
        elif len([key for key in keys if 'node' in key]) > 0:
            return True
        else:
            return False

    @property
    def num_nodes(self) -> Optional[int]:
        # We sequentially access attributes that reveal the number of nodes.
        if 'num_nodes' in self:
            return self['num_nodes']
        for key, value in self.items():
            if isinstance(value, Tensor) and key in N_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.size(cat_dim)
            if isinstance(value, np.ndarray) and key in N_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, TensorFrame) and key in N_KEYS:
                return value.num_rows
        for key, value in self.items():
            if isinstance(value, Tensor) and 'node' in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.size(cat_dim)
            if isinstance(value, np.ndarray) and 'node' in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, TensorFrame) and 'node' in key:
                return value.num_rows
        if 'adj' in self and isinstance(self.adj, SparseTensor):
            return self.adj.size(0)
        if 'adj_t' in self and isinstance(self.adj_t, SparseTensor):
            return self.adj_t.size(1)
        warnings.warn(
            f"Unable to accurately infer 'num_nodes' from the attribute set "
            f"'{set(self.keys())}'. Please explicitly set 'num_nodes' as an "
            f"attribute of " +
            ("'data'" if self._key is None else f"'data[{self._key}]'") +
            " to suppress this warning")
        if 'edge_index' in self and isinstance(self.edge_index, Tensor):
            if self.edge_index.numel() > 0:
                return int(self.edge_index.max()) + 1
            else:
                return 0
        if 'face' in self and isinstance(self.face, Tensor):
            if self.face.numel() > 0:
                return int(self.face.max()) + 1
            else:
                return 0
        return None

    @property
    def num_node_features(self) -> int:
        if 'x' in self and isinstance(self.x, Tensor):
            return 1 if self.x.dim() == 1 else self.x.size(-1)
        if 'x' in self and isinstance(self.x, np.ndarray):
            return 1 if self.x.ndim == 1 else self.x.shape[-1]
        if 'x' in self and isinstance(self.x, SparseTensor):
            return 1 if self.x.dim() == 1 else self.x.size(-1)
        if 'x' in self and isinstance(self.x, TensorFrame):
            return self.x.num_cols
        if 'tf' in self and isinstance(self.tf, TensorFrame):
            return self.tf.num_cols
        return 0

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def is_node_attr(self, key: str) -> bool:
        if '_cached_attr' not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.NODE]:
            return True
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if (isinstance(value, (list, tuple, TensorFrame))
                and len(value) == self.num_nodes):
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if not isinstance(value, (Tensor, np.ndarray)):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        if value.shape[cat_dim] != self.num_nodes:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        self._cached_attr[AttrType.NODE].add(key)
        return True

    def is_edge_attr(self, key: str) -> bool:
        return False

    def node_attrs(self) -> List[str]:
        return [key for key in self.keys() if self.is_node_attr(key)]


class EdgeStorage(BaseStorage):
    r"""We support multiple ways to store edge connectivity in a
    :class:`EdgeStorage` object:

    * :obj:`edge_index`: A :class:`torch.LongTensor` holding edge indices in
      COO format with shape :obj:`[2, num_edges]` (the default format)

    * :obj:`adj`: A :class:`torch_sparse.SparseTensor` holding edge indices in
      a sparse format, supporting both COO and CSR format.

    * :obj:`adj_t`: A **transposed** :class:`torch_sparse.SparseTensor` holding
      edge indices in a sparse format, supporting both COO and CSR format.
      This is the most efficient one for graph-based deep learning models as
      indices are sorted based on target nodes.
    """
    @property
    def _key(self) -> EdgeType:
        key = self.__dict__.get('_key', None)
        if key is None or not isinstance(key, tuple) or not len(key) == 3:
            raise ValueError("'_key' does not denote a valid edge type")
        return key

    @property
    def edge_index(self) -> Tensor:
        if 'edge_index' in self:
            return self['edge_index']
        if 'adj' in self and isinstance(self.adj, SparseTensor):
            return torch.stack(self.adj.coo()[:2], dim=0)
        if 'adj_t' in self and isinstance(self.adj_t, SparseTensor):
            return torch.stack(self.adj_t.coo()[:2][::-1], dim=0)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute "
            f"'edge_index', 'adj' or 'adj_t'")

    @property
    def num_edges(self) -> int:
        # We sequentially access attributes that reveal the number of edges.
        if 'num_edges' in self:
            return self['num_edges']
        for key, value in self.items():
            if isinstance(value, Tensor) and key in E_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.size(cat_dim)
            if isinstance(value, Tensor) and key in E_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, TensorFrame) and key in E_KEYS:
                return value.num_rows
        for key, value in self.items():
            if isinstance(value, Tensor) and 'edge' in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.size(cat_dim)
            if isinstance(value, np.ndarray) and 'edge' in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, TensorFrame) and 'edge' in key:
                return value.num_rows
        for value in self.values('adj', 'adj_t'):
            if isinstance(value, SparseTensor):
                return value.nnz()
            elif is_torch_sparse_tensor(value):
                return value._nnz()
        return 0

    @property
    def num_edge_features(self) -> int:
        if 'edge_attr' in self and isinstance(self.edge_attr, Tensor):
            return 1 if self.edge_attr.dim() == 1 else self.edge_attr.size(-1)
        if 'edge_attr' in self and isinstance(self.edge_attr, np.ndarray):
            return 1 if self.edge_attr.ndim == 1 else self.edge_attr.shape[-1]
        if 'edge_attr' in self and isinstance(self.edge_attr, TensorFrame):
            return self.edge_attr.num_cols
        return 0

    @property
    def num_features(self) -> int:
        return self.num_edge_features

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:

        if self._key is None:
            raise NameError("Unable to infer 'size' without explicit "
                            "'_key' assignment")

        size = (self._parent()[self._key[0]].num_nodes,
                self._parent()[self._key[-1]].num_nodes)

        return size if dim is None else size[dim]

    def is_node_attr(self, key: str) -> bool:
        return False

    def is_edge_attr(self, key: str) -> bool:
        if '_cached_attr' not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.EDGE]:
            return True
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if (isinstance(value, (list, tuple, TensorFrame))
                and len(value) == self.num_edges):
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if not isinstance(value, (Tensor, np.ndarray)):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        if value.shape[cat_dim] != self.num_edges:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        self._cached_attr[AttrType.EDGE].add(key)
        return True

    def edge_attrs(self) -> List[str]:
        return [key for key in self.keys() if self.is_edge_attr(key)]

    def is_sorted(self, sort_by_row: bool = True) -> bool:
        if 'edge_index' in self:
            index = self.edge_index[0] if sort_by_row else self.edge_index[1]
            return bool(torch.all(index[:-1] <= index[1:]))
        return True

    def sort(self, sort_by_row: bool = True) -> 'EdgeStorage':
        if 'edge_index' in self:
            edge_attrs = self.edge_attrs()
            edge_attrs.remove('edge_index')
            edge_feats = [self[edge_attr] for edge_attr in edge_attrs]
            self.edge_index, edge_feats = sort_edge_index(
                self.edge_index, edge_feats, sort_by_row=sort_by_row)
            for key, edge_feat in zip(edge_attrs, edge_feats):
                self[key] = edge_feat
        return self

    def is_coalesced(self) -> bool:
        for value in self.values('adj', 'adj_t'):
            return value.is_coalesced()

        if 'edge_index' in self:
            new_edge_index = coalesce(
                self.edge_index,
                num_nodes=max(self.size(0), self.size(1)),
            )
            return (self.edge_index.numel() == new_edge_index.numel()
                    and torch.equal(self.edge_index, new_edge_index))

        return True

    def coalesce(self, reduce: str = 'sum') -> 'EdgeStorage':
        for key, value in self.items('adj', 'adj_t'):
            self[key] = value.coalesce(reduce)

        if 'edge_index' in self:
            if 'edge_attr' in self:
                self.edge_index, self.edge_attr = coalesce(
                    self.edge_index,
                    self.edge_attr,
                    num_nodes=max(self.size(0), self.size(1)),
                )
            else:
                self.edge_index = coalesce(
                    self.edge_index,
                    num_nodes=max(self.size(0), self.size(1)),
                )

        return self

    def has_isolated_nodes(self) -> bool:
        edge_index, num_nodes = self.edge_index, self.size(1)
        if num_nodes is None:
            raise NameError("Unable to infer 'num_nodes'")
        if self.is_bipartite():
            return torch.unique(edge_index[1]).numel() < num_nodes
        else:
            return contains_isolated_nodes(edge_index, num_nodes)

    def has_self_loops(self) -> bool:
        if self.is_bipartite():
            return False
        edge_index = self.edge_index
        return int((edge_index[0] == edge_index[1]).sum()) > 0

    def is_undirected(self) -> bool:
        if self.is_bipartite():
            return False

        for value in self.values('adj', 'adj_t'):
            return value.is_symmetric()

        edge_index = self.edge_index
        edge_attr = self.edge_attr if 'edge_attr' in self else None
        return is_undirected(edge_index, edge_attr, num_nodes=self.size(0))

    def is_directed(self) -> bool:
        return not self.is_undirected()

    def is_bipartite(self) -> bool:
        return self._key is not None and self._key[0] != self._key[-1]


class GlobalStorage(NodeStorage, EdgeStorage):
    @property
    def _key(self) -> Any:
        return None

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

    def is_node_attr(self, key: str) -> bool:
        if '_cached_attr' not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.NODE]:
            return True
        if key in self._cached_attr[AttrType.EDGE]:
            return False
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if (isinstance(value, (list, tuple, TensorFrame))
                and len(value) == self.num_nodes):
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if not isinstance(value, (Tensor, np.ndarray)):
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        num_nodes, num_edges = self.num_nodes, self.num_edges

        if value.shape[cat_dim] != num_nodes:
            if value.shape[cat_dim] == num_edges:
                self._cached_attr[AttrType.EDGE].add(key)
            else:
                self._cached_attr[AttrType.OTHER].add(key)
            return False

        if num_nodes != num_edges:
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if 'edge' not in key:
            self._cached_attr[AttrType.NODE].add(key)
            return True
        else:
            self._cached_attr[AttrType.EDGE].add(key)
            return False

    def is_edge_attr(self, key: str) -> bool:
        if '_cached_attr' not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.EDGE]:
            return True
        if key in self._cached_attr[AttrType.NODE]:
            return False
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if (isinstance(value, (list, tuple, TensorFrame))
                and len(value) == self.num_edges):
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if not isinstance(value, (Tensor, np.ndarray)):
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        num_nodes, num_edges = self.num_nodes, self.num_edges

        if value.shape[cat_dim] != num_edges:
            if value.shape[cat_dim] == num_nodes:
                self._cached_attr[AttrType.NODE].add(key)
            else:
                self._cached_attr[AttrType.OTHER].add(key)
            return False

        if num_edges != num_nodes:
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if 'edge' in key:
            self._cached_attr[AttrType.EDGE].add(key)
            return True
        else:
            self._cached_attr[AttrType.NODE].add(key)
            return False


def recursive_apply_(data: Any, func: Callable):
    if isinstance(data, Tensor):
        func(data)
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        for value in data:
            recursive_apply_(value, func)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        for value in data:
            recursive_apply_(value, func)
    elif isinstance(data, Mapping):
        for value in data.values():
            recursive_apply_(value, func)
    else:
        try:
            func(data)
        except:  # noqa
            pass


def recursive_apply(data: Any, func: Callable) -> Any:
    if isinstance(data, Tensor):
        return func(data)
    elif isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return func(data)
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(recursive_apply(d, func) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, func) for d in data]
    elif isinstance(data, Mapping):
        return {key: recursive_apply(data[key], func) for key in data}
    else:
        try:
            return func(data)
        except:  # noqa
            return data
