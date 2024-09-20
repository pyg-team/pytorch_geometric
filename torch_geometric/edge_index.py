import functools
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    overload,
)

import torch
import torch.utils._pytree as pytree
from torch import Tensor

import torch_geometric.typing
from torch_geometric import Index, is_compiling
from torch_geometric.index import index2ptr, ptr2index
from torch_geometric.typing import INDEX_DTYPES, SparseTensor

aten = torch.ops.aten

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}

ReduceType = Literal['sum', 'mean', 'amin', 'amax', 'add', 'min', 'max']
PYG_REDUCE: Dict[ReduceType, ReduceType] = {
    'add': 'sum',
    'amin': 'min',
    'amax': 'max'
}
TORCH_REDUCE: Dict[ReduceType, ReduceType] = {
    'add': 'sum',
    'min': 'amin',
    'max': 'amax'
}


class SortOrder(Enum):
    ROW = 'row'
    COL = 'col'


class CatMetadata(NamedTuple):
    nnz: List[int]
    sparse_size: List[Tuple[Optional[int], Optional[int]]]
    sort_order: List[Optional[SortOrder]]
    is_undirected: List[bool]


def implements(torch_function: Callable) -> Callable:
    r"""Registers a :pytorch:`PyTorch` function override."""
    @functools.wraps(torch_function)
    def decorator(my_function: Callable) -> Callable:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


def set_tuple_item(
    values: Tuple[Any, ...],
    dim: int,
    value: Any,
) -> Tuple[Any, ...]:
    if dim < -len(values) or dim >= len(values):
        raise IndexError("tuple index out of range")

    dim = dim + len(values) if dim < 0 else dim
    return values[:dim] + (value, ) + values[dim + 1:]


def maybe_add(
    value: Sequence[Optional[int]],
    other: Union[int, Sequence[Optional[int]]],
    alpha: int = 1,
) -> Tuple[Optional[int], ...]:

    if isinstance(other, int):
        return tuple(v + alpha * other if v is not None else None
                     for v in value)

    assert len(value) == len(other)
    return tuple(v + alpha * o if v is not None and o is not None else None
                 for v, o in zip(value, other))


def maybe_sub(
    value: Sequence[Optional[int]],
    other: Union[int, Sequence[Optional[int]]],
    alpha: int = 1,
) -> Tuple[Optional[int], ...]:

    if isinstance(other, int):
        return tuple(v - alpha * other if v is not None else None
                     for v in value)

    assert len(value) == len(other)
    return tuple(v - alpha * o if v is not None and o is not None else None
                 for v, o in zip(value, other))


def assert_valid_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in INDEX_DTYPES:
        raise ValueError(f"'EdgeIndex' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{INDEX_DTYPES})")


def assert_two_dimensional(tensor: Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"'EdgeIndex' needs to be two-dimensional "
                         f"(got {tensor.dim()} dimensions)")
    if not torch.jit.is_tracing() and tensor.size(0) != 2:
        raise ValueError(f"'EdgeIndex' needs to have a shape of "
                         f"[2, *] (got {list(tensor.size())})")


def assert_contiguous(tensor: Tensor) -> None:
    if not tensor[0].is_contiguous() or not tensor[1].is_contiguous():
        raise ValueError("'EdgeIndex' needs to be contiguous. Please call "
                         "`edge_index.contiguous()` before proceeding.")


def assert_symmetric(size: Tuple[Optional[int], Optional[int]]) -> None:
    if (not torch.jit.is_tracing() and size[0] is not None
            and size[1] is not None and size[0] != size[1]):
        raise ValueError(f"'EdgeIndex' is undirected but received a "
                         f"non-symmetric size (got {list(size)})")


def assert_sorted(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: 'EdgeIndex', *args: Any, **kwargs: Any) -> Any:
        if not self.is_sorted:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"Cannot call '{func.__name__}' since '{cls_name}' is not "
                f"sorted. Please call `{cls_name}.sort_by(...)` first.")
        return func(self, *args, **kwargs)

    return wrapper


class EdgeIndex(Tensor):
    r"""A COO :obj:`edge_index` tensor with additional (meta)data attached.

    :class:`EdgeIndex` is a :pytorch:`null` :class:`torch.Tensor`, that holds
    an :obj:`edge_index` representation of shape :obj:`[2, num_edges]`.
    Edges are given as pairwise source and destination node indices in sparse
    COO format.

    While :class:`EdgeIndex` sub-classes a general :pytorch:`null`
    :class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

    * :obj:`sparse_size`: The underlying sparse matrix size
    * :obj:`sort_order`: The sort order (if present), either by row or column.
    * :obj:`is_undirected`: Whether edges are bidirectional.

    Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
    in case its representation is sorted, such as its :obj:`rowptr` or
    :obj:`colptr`, or the permutation vector for going from CSR to CSC or vice
    versa.
    Caches are filled based on demand (*e.g.*, when calling
    :meth:`EdgeIndex.sort_by`), or when explicitly requested via
    :meth:`EdgeIndex.fill_cache_`, and are maintained and adjusted over its
    lifespan (*e.g.*, when calling :meth:`EdgeIndex.flip`).

    This representation ensures optimal computation in GNN message passing
    schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
    workflows.

    .. code-block:: python

        from torch_geometric import EdgeIndex

        edge_index = EdgeIndex(
            [[0, 1, 1, 2],
             [1, 0, 2, 1]]
            sparse_size=(3, 3),
            sort_order='row',
            is_undirected=True,
            device='cpu',
        )
        >>> EdgeIndex([[0, 1, 1, 2],
        ...            [1, 0, 2, 1]])
        assert edge_index.is_sorted_by_row
        assert edge_index.is_undirected

        # Flipping order:
        edge_index = edge_index.flip(0)
        >>> EdgeIndex([[1, 0, 2, 1],
        ...            [0, 1, 1, 2]])
        assert edge_index.is_sorted_by_col
        assert edge_index.is_undirected

        # Filtering:
        mask = torch.tensor([True, True, True, False])
        edge_index = edge_index[:, mask]
        >>> EdgeIndex([[1, 0, 2],
        ...            [0, 1, 1]])
        assert edge_index.is_sorted_by_col
        assert not edge_index.is_undirected

        # Sparse-Dense Matrix Multiplication:
        out = edge_index.flip(0) @ torch.randn(3, 16)
        assert out.size() == (3, 16)
    """
    # See "https://pytorch.org/docs/stable/notes/extending.html"
    # for a basic tutorial on how to subclass `torch.Tensor`.

    # The underlying tensor representation:
    _data: Tensor

    # The size of the underlying sparse matrix:
    _sparse_size: Tuple[Optional[int], Optional[int]] = (None, None)

    # Whether the `edge_index` representation is non-sorted (`None`), or sorted
    # based on row or column values.
    _sort_order: Optional[SortOrder] = None

    # Whether the `edge_index` is undirected:
    # NOTE `is_undirected` allows us to assume symmetric adjacency matrix size
    # and to share compressed pointer representations, however, it does not
    # allow us get rid of CSR/CSC permutation vectors since ordering within
    # neighborhoods is not necessarily deterministic.
    _is_undirected: bool = False

    # A cache for its compressed representation:
    _indptr: Optional[Tensor] = None

    # A cache for its transposed representation:
    _T_perm: Optional[Tensor] = None
    _T_index: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None)
    _T_indptr: Optional[Tensor] = None

    # A cached "1"-value vector for `torch.sparse` matrix multiplication:
    _value: Optional[Tensor] = None

    # Whenever we perform a concatenation of edge indices, we cache the
    # original metadata to be able to reconstruct individual edge indices:
    _cat_metadata: Optional[CatMetadata] = None

    @staticmethod
    def __new__(
        cls: Type,
        data: Any,
        *args: Any,
        sparse_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
        sort_order: Optional[Union[str, SortOrder]] = None,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> 'EdgeIndex':
        if not isinstance(data, Tensor):
            data = torch.tensor(data, *args, **kwargs)
        elif len(args) > 0:
            raise TypeError(
                f"new() received an invalid combination of arguments - got "
                f"(Tensor, {', '.join(str(type(arg)) for arg in args)})")
        elif len(kwargs) > 0:
            raise TypeError(f"new() received invalid keyword arguments - got "
                            f"{set(kwargs.keys())})")

        assert isinstance(data, Tensor)

        indptr: Optional[Tensor] = None

        if isinstance(data, cls):  # If passed `EdgeIndex`, inherit metadata:
            indptr = data._indptr
            sparse_size = sparse_size or data.sparse_size()
            sort_order = sort_order or data.sort_order
            is_undirected = is_undirected or data.is_undirected

        # Convert `torch.sparse` tensors to `EdgeIndex` representation:
        if data.layout == torch.sparse_coo:
            sort_order = SortOrder.ROW
            sparse_size = sparse_size or (data.size(0), data.size(1))
            data = data.indices()

        if data.layout == torch.sparse_csr:
            indptr = data.crow_indices()
            col = data.col_indices()

            assert isinstance(indptr, Tensor)
            row = ptr2index(indptr, output_size=col.numel())

            sort_order = SortOrder.ROW
            sparse_size = sparse_size or (data.size(0), data.size(1))
            if sparse_size[0] is not None and sparse_size[0] != data.size(0):
                indptr = None
            data = torch.stack([row, col], dim=0)

        if (torch_geometric.typing.WITH_PT112
                and data.layout == torch.sparse_csc):
            row = data.row_indices()
            indptr = data.ccol_indices()

            assert isinstance(indptr, Tensor)
            col = ptr2index(indptr, output_size=row.numel())

            sort_order = SortOrder.COL
            sparse_size = sparse_size or (data.size(0), data.size(1))
            if sparse_size[1] is not None and sparse_size[1] != data.size(1):
                indptr = None
            data = torch.stack([row, col], dim=0)

        assert_valid_dtype(data)
        assert_two_dimensional(data)
        assert_contiguous(data)

        if sparse_size is None:
            sparse_size = (None, None)

        if is_undirected:
            assert_symmetric(sparse_size)
            if sparse_size[0] is not None and sparse_size[1] is None:
                sparse_size = (sparse_size[0], sparse_size[0])
            elif sparse_size[0] is None and sparse_size[1] is not None:
                sparse_size = (sparse_size[1], sparse_size[1])

        out = Tensor._make_wrapper_subclass(  # type: ignore
            cls,
            size=data.size(),
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            layout=data.layout,
            requires_grad=False,
        )
        assert isinstance(out, EdgeIndex)

        # Attach metadata:
        out._data = data
        out._sparse_size = sparse_size
        out._sort_order = None if sort_order is None else SortOrder(sort_order)
        out._is_undirected = is_undirected
        out._indptr = indptr

        if isinstance(data, cls):  # If passed `EdgeIndex`, inherit metadata:
            out._data = data._data
            out._T_perm = data._T_perm
            out._T_index = data._T_index
            out._T_indptr = data._T_indptr
            out._value = out._value

            # Reset metadata if cache is invalidated:
            num_rows = sparse_size[0]
            if num_rows is not None and num_rows != data.sparse_size(0):
                out._indptr = None

            num_cols = sparse_size[1]
            if num_cols is not None and num_cols != data.sparse_size(1):
                out._T_indptr = None

        return out

    # Validation ##############################################################

    def validate(self) -> 'EdgeIndex':
        r"""Validates the :class:`EdgeIndex` representation.

        In particular, it ensures that

        * it only holds valid indices.
        * the sort order is correctly set.
        * indices are bidirectional in case it is specified as undirected.
        """
        assert_valid_dtype(self._data)
        assert_two_dimensional(self._data)
        assert_contiguous(self._data)
        if self.is_undirected:
            assert_symmetric(self.sparse_size())

        if self.numel() > 0 and self._data.min() < 0:
            raise ValueError(f"'{self.__class__.__name__}' contains negative "
                             f"indices (got {int(self.min())})")

        if (self.numel() > 0 and self.num_rows is not None
                and self._data[0].max() >= self.num_rows):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its number of rows "
                             f"(got {int(self._data[0].max())}, but expected "
                             f"values smaller than {self.num_rows})")

        if (self.numel() > 0 and self.num_cols is not None
                and self._data[1].max() >= self.num_cols):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its number of columns "
                             f"(got {int(self._data[1].max())}, but expected "
                             f"values smaller than {self.num_cols})")

        if self.is_sorted_by_row and (self._data[0].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"row indices")

        if self.is_sorted_by_col and (self._data[1].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"column indices")

        if self.is_undirected:
            flat_index1 = self._data[0] * self.get_num_rows() + self._data[1]
            flat_index1 = flat_index1.sort()[0]
            flat_index2 = self._data[1] * self.get_num_cols() + self._data[0]
            flat_index2 = flat_index2.sort()[0]
            if not torch.equal(flat_index1, flat_index2):
                raise ValueError(f"'{self.__class__.__name__}' is not "
                                 f"undirected")

        return self

    # Properties ##############################################################

    @overload
    def sparse_size(self) -> Tuple[Optional[int], Optional[int]]:
        pass

    @overload
    def sparse_size(self, dim: int) -> Optional[int]:
        pass

    def sparse_size(
        self,
        dim: Optional[int] = None,
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""The size of the underlying sparse matrix.
        If :obj:`dim` is specified, returns an integer holding the size of that
        sparse dimension.

        Args:
            dim (int, optional): The dimension for which to retrieve the size.
                (default: :obj:`None`)
        """
        if dim is not None:
            return self._sparse_size[dim]
        return self._sparse_size

    @property
    def num_rows(self) -> Optional[int]:
        r"""The number of rows of the underlying sparse matrix."""
        return self._sparse_size[0]

    @property
    def num_cols(self) -> Optional[int]:
        r"""The number of columns of the underlying sparse matrix."""
        return self._sparse_size[1]

    @property
    def sort_order(self) -> Optional[str]:
        r"""The sort order of indices, either :obj:`"row"`, :obj:`"col"` or
        :obj:`None`.
        """
        return None if self._sort_order is None else self._sort_order.value

    @property
    def is_sorted(self) -> bool:
        r"""Returns whether indices are either sorted by rows or columns."""
        return self._sort_order is not None

    @property
    def is_sorted_by_row(self) -> bool:
        r"""Returns whether indices are sorted by rows."""
        return self._sort_order == SortOrder.ROW

    @property
    def is_sorted_by_col(self) -> bool:
        r"""Returns whether indices are sorted by columns."""
        return self._sort_order == SortOrder.COL

    @property
    def is_undirected(self) -> bool:
        r"""Returns whether indices are bidirectional."""
        return self._is_undirected

    @property
    def dtype(self) -> torch.dtype:  # type: ignore
        # TODO Remove once PyTorch does not override `dtype` in `DataLoader`.
        return self._data.dtype

    # Cache Interface #########################################################

    @overload
    def get_sparse_size(self) -> torch.Size:
        pass

    @overload
    def get_sparse_size(self, dim: int) -> int:
        pass

    def get_sparse_size(
        self,
        dim: Optional[int] = None,
    ) -> Union[torch.Size, int]:
        r"""The size of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        If :obj:`dim` is specified, returns an integer holding the size of that
        sparse dimension.

        Args:
            dim (int, optional): The dimension for which to retrieve the size.
                (default: :obj:`None`)
        """
        if dim is not None:
            size = self._sparse_size[dim]
            if size is not None:
                return size

            if self.is_undirected:
                size = int(self._data.max()) + 1 if self.numel() > 0 else 0
                self._sparse_size = (size, size)
                return size

            size = int(self._data[dim].max()) + 1 if self.numel() > 0 else 0
            self._sparse_size = set_tuple_item(self._sparse_size, dim, size)
            return size

        return torch.Size((self.get_sparse_size(0), self.get_sparse_size(1)))

    def sparse_resize_(  # type: ignore
        self,
        num_rows: Optional[int],
        num_cols: Optional[int],
    ) -> 'EdgeIndex':
        r"""Assigns or re-assigns the size of the underlying sparse matrix.

        Args:
            num_rows (int, optional): The number of rows.
            num_cols (int, optional): The number of columns.
        """
        if self.is_undirected:
            if num_rows is not None and num_cols is None:
                num_cols = num_rows
            elif num_cols is not None and num_rows is None:
                num_rows = num_cols

            if num_rows is not None and num_rows != num_cols:
                raise ValueError(f"'EdgeIndex' is undirected but received a "
                                 f"non-symmetric size "
                                 f"(got [{num_rows}, {num_cols}])")

        def _modify_ptr(
            ptr: Optional[Tensor],
            size: Optional[int],
        ) -> Optional[Tensor]:

            if ptr is None or size is None:
                return None

            if ptr.numel() - 1 >= size:
                return ptr[:size + 1]

            fill_value = ptr.new_full(
                (size - ptr.numel() + 1, ),
                fill_value=ptr[-1],  # type: ignore
            )
            return torch.cat([ptr, fill_value], dim=0)

        if self.is_sorted_by_row:
            self._indptr = _modify_ptr(self._indptr, num_rows)
            self._T_indptr = _modify_ptr(self._T_indptr, num_cols)

        if self.is_sorted_by_col:
            self._indptr = _modify_ptr(self._indptr, num_cols)
            self._T_indptr = _modify_ptr(self._T_indptr, num_rows)

        self._sparse_size = (num_rows, num_cols)

        return self

    def get_num_rows(self) -> int:
        r"""The number of rows of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        return self.get_sparse_size(0)

    def get_num_cols(self) -> int:
        r"""The number of columns of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        return self.get_sparse_size(1)

    @assert_sorted
    def get_indptr(self) -> Tensor:
        r"""Returns the compressed index representation in case
        :class:`EdgeIndex` is sorted.
        """
        if self._indptr is not None:
            return self._indptr

        if self.is_undirected and self._T_indptr is not None:
            return self._T_indptr

        dim = 0 if self.is_sorted_by_row else 1
        self._indptr = index2ptr(self._data[dim], self.get_sparse_size(dim))

        return self._indptr

    @assert_sorted
    def _sort_by_transpose(self) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        from torch_geometric.utils import index_sort

        dim = 1 if self.is_sorted_by_row else 0

        if self._T_perm is None:
            max_index = self.get_sparse_size(dim)
            index, perm = index_sort(self._data[dim], max_index)
            self._T_index = set_tuple_item(self._T_index, dim, index)
            self._T_perm = perm.to(self.dtype)

        if self._T_index[1 - dim] is None:
            self._T_index = set_tuple_item(  #
                self._T_index, 1 - dim, self._data[1 - dim][self._T_perm])

        row, col = self._T_index
        assert row is not None and col is not None

        return (row, col), self._T_perm

    @assert_sorted
    def get_csr(self) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        r"""Returns the compressed CSR representation
        :obj:`(rowptr, col), perm` in case :class:`EdgeIndex` is sorted.
        """
        if self.is_sorted_by_row:
            return (self.get_indptr(), self._data[1]), None

        assert self.is_sorted_by_col
        (row, col), perm = self._sort_by_transpose()

        if self._T_indptr is not None:
            rowptr = self._T_indptr
        elif self.is_undirected and self._indptr is not None:
            rowptr = self._indptr
        else:
            rowptr = self._T_indptr = index2ptr(row, self.get_num_rows())

        return (rowptr, col), perm

    @assert_sorted
    def get_csc(self) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        r"""Returns the compressed CSC representation
        :obj:`(colptr, row), perm` in case :class:`EdgeIndex` is sorted.
        """
        if self.is_sorted_by_col:
            return (self.get_indptr(), self._data[0]), None

        assert self.is_sorted_by_row
        (row, col), perm = self._sort_by_transpose()

        if self._T_indptr is not None:
            colptr = self._T_indptr
        elif self.is_undirected and self._indptr is not None:
            colptr = self._indptr
        else:
            colptr = self._T_indptr = index2ptr(col, self.get_num_cols())

        return (colptr, row), perm

    def _get_value(self, dtype: Optional[torch.dtype] = None) -> Tensor:
        if self._value is not None:
            if (dtype or torch.get_default_dtype()) == self._value.dtype:
                return self._value

        # Expanded tensors are not yet supported in all PyTorch code paths :(
        # value = torch.ones(1, dtype=dtype, device=self.device)
        # value = value.expand(self.size(1))
        self._value = torch.ones(self.size(1), dtype=dtype, device=self.device)

        return self._value

    def fill_cache_(self, no_transpose: bool = False) -> 'EdgeIndex':
        r"""Fills the cache with (meta)data information.

        Args:
            no_transpose (bool, optional): If set to :obj:`True`, will not fill
                the cache with information about the transposed
                :class:`EdgeIndex`. (default: :obj:`False`)
        """
        self.get_sparse_size()

        if self.is_sorted_by_row:
            self.get_csr()
            if not no_transpose:
                self.get_csc()
        elif self.is_sorted_by_col:
            self.get_csc()
            if not no_transpose:
                self.get_csr()

        return self

    # Methods #################################################################

    def share_memory_(self) -> 'EdgeIndex':
        """"""  # noqa: D419
        self._data.share_memory_()
        if self._indptr is not None:
            self._indptr.share_memory_()
        if self._T_perm is not None:
            self._T_perm.share_memory_()
        if self._T_index[0] is not None:
            self._T_index[0].share_memory_()
        if self._T_index[1] is not None:
            self._T_index[1].share_memory_()
        if self._T_indptr is not None:
            self._T_indptr.share_memory_()
        if self._value is not None:
            self._value.share_memory_()
        return self

    def is_shared(self) -> bool:
        """"""  # noqa: D419
        return self._data.is_shared()

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`EdgeIndex` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self._data

    def sort_by(
        self,
        sort_order: Union[str, SortOrder],
        stable: bool = False,
    ) -> 'SortReturnType':
        r"""Sorts the elements by row or column indices.

        Args:
            sort_order (str): The sort order, either :obj:`"row"` or
                :obj:`"col"`.
            stable (bool, optional): Makes the sorting routine stable, which
                guarantees that the order of equivalent elements is preserved.
                (default: :obj:`False`)
        """
        from torch_geometric.utils import index_sort

        sort_order = SortOrder(sort_order)

        if self._sort_order == sort_order:  # Nothing to do.
            return SortReturnType(self, None)

        if self.is_sorted:
            (row, col), perm = self._sort_by_transpose()
            edge_index = torch.stack([row, col], dim=0)

        # Otherwise, perform sorting:
        elif sort_order == SortOrder.ROW:
            row, perm = index_sort(self._data[0], self.get_num_rows(), stable)
            edge_index = torch.stack([row, self._data[1][perm]], dim=0)

        else:
            col, perm = index_sort(self._data[1], self.get_num_cols(), stable)
            edge_index = torch.stack([self._data[0][perm], col], dim=0)

        out = self.__class__(edge_index)

        # We can inherit metadata and (mostly) cache:
        out._sparse_size = self.sparse_size()
        out._sort_order = sort_order
        out._is_undirected = self.is_undirected

        out._indptr = self._indptr
        out._T_indptr = self._T_indptr

        # NOTE We cannot copy CSR<>CSC permutations since we don't require that
        # local neighborhoods are sorted, and thus they may run out of sync.

        out._value = self._value

        return SortReturnType(out, perm)

    def to_dense(  # type: ignore
        self,
        value: Optional[Tensor] = None,
        fill_value: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a dense :class:`torch.Tensor`.

        .. warning::

            In case of duplicated edges, the behavior is non-deterministic (one
            of the values from :obj:`value` will be picked arbitrarily). For
            deterministic behavior, consider calling
            :meth:`~torch_geometric.utils.coalesce` beforehand.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
            fill_value (float, optional): The fill value for remaining elements
                in the dense matrix. (default: :obj:`0.0`)
            dtype (torch.dtype, optional): The data type of the returned
                tensor. (default: :obj:`None`)
        """
        dtype = value.dtype if value is not None else dtype

        size = self.get_sparse_size()
        if value is not None and value.dim() > 1:
            size = size + value.size()[1:]  # type: ignore

        out = torch.full(size, fill_value, dtype=dtype, device=self.device)
        out[self._data[0], self._data[1]] = value if value is not None else 1

        return out

    def to_sparse_coo(self, value: Optional[Tensor] = None) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a :pytorch:`null`
        :class:`torch.sparse_coo_tensor`.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        value = self._get_value() if value is None else value

        if not torch_geometric.typing.WITH_PT21:
            out = torch.sparse_coo_tensor(
                indices=self._data,
                values=value,
                size=self.get_sparse_size(),
                device=self.device,
                requires_grad=value.requires_grad,
            )
            if self.is_sorted_by_row:
                out = out._coalesced_(True)
            return out

        return torch.sparse_coo_tensor(
            indices=self._data,
            values=value,
            size=self.get_sparse_size(),
            device=self.device,
            requires_grad=value.requires_grad,
            is_coalesced=True if self.is_sorted_by_row else None,
        )

    def to_sparse_csr(  # type: ignore
            self,
            value: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a :pytorch:`null`
        :class:`torch.sparse_csr_tensor`.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        (rowptr, col), perm = self.get_csr()
        if value is not None and perm is not None:
            value = value[perm]
        elif value is None:
            value = self._get_value()

        return torch.sparse_csr_tensor(
            crow_indices=rowptr,
            col_indices=col,
            values=value,
            size=self.get_sparse_size(),
            device=self.device,
            requires_grad=value.requires_grad,
        )

    def to_sparse_csc(  # type: ignore
            self,
            value: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a :pytorch:`null`
        :class:`torch.sparse_csc_tensor`.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        if not torch_geometric.typing.WITH_PT112:
            raise NotImplementedError(
                "'to_sparse_csc' not supported for PyTorch < 1.12")

        (colptr, row), perm = self.get_csc()
        if value is not None and perm is not None:
            value = value[perm]
        elif value is None:
            value = self._get_value()

        return torch.sparse_csc_tensor(
            ccol_indices=colptr,
            row_indices=row,
            values=value,
            size=self.get_sparse_size(),
            device=self.device,
            requires_grad=value.requires_grad,
        )

    def to_sparse(  # type: ignore
        self,
        *,
        layout: torch.layout = torch.sparse_coo,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a
        :pytorch:`null` :class:`torch.sparse` tensor.

        Args:
            layout (torch.layout, optional): The desired sparse layout. One of
                :obj:`torch.sparse_coo`, :obj:`torch.sparse_csr`, or
                :obj:`torch.sparse_csc`. (default: :obj:`torch.sparse_coo`)
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        if layout is None or layout == torch.sparse_coo:
            return self.to_sparse_coo(value)
        if layout == torch.sparse_csr:
            return self.to_sparse_csr(value)
        if torch_geometric.typing.WITH_PT112 and layout == torch.sparse_csc:
            return self.to_sparse_csc(value)

        raise ValueError(f"Unexpected tensor layout (got '{layout}')")

    def to_sparse_tensor(
        self,
        value: Optional[Tensor] = None,
    ) -> SparseTensor:
        r"""Converts :class:`EdgeIndex` into a
        :class:`torch_sparse.SparseTensor`.
        Requires that :obj:`torch-sparse` is installed.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                (default: :obj:`None`)
        """
        return SparseTensor(
            row=self._data[0],
            col=self._data[1],
            rowptr=self._indptr if self.is_sorted_by_row else None,
            value=value,
            sparse_sizes=self.get_sparse_size(),
            is_sorted=self.is_sorted_by_row,
            trust_data=True,
        )

    # TODO Investigate how to avoid overlapping return types here.
    @overload
    def matmul(  # type: ignore
        self,
        other: 'EdgeIndex',
        input_value: Optional[Tensor] = None,
        other_value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
        transpose: bool = False,
    ) -> Tuple['EdgeIndex', Tensor]:
        pass

    @overload
    def matmul(
        self,
        other: Tensor,
        input_value: Optional[Tensor] = None,
        other_value: None = None,
        reduce: ReduceType = 'sum',
        transpose: bool = False,
    ) -> Tensor:
        pass

    def matmul(
        self,
        other: Union[Tensor, 'EdgeIndex'],
        input_value: Optional[Tensor] = None,
        other_value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
        transpose: bool = False,
    ) -> Union[Tensor, Tuple['EdgeIndex', Tensor]]:
        r"""Performs a matrix multiplication of the matrices :obj:`input` and
        :obj:`other`.
        If :obj:`input` is a :math:`(n \times m)` matrix and :obj:`other` is a
        :math:`(m \times p)` tensor, then the output will be a
        :math:`(n \times p)` tensor.
        See :meth:`torch.matmul` for more information.

        :obj:`input` is a sparse matrix as denoted by the indices in
        :class:`EdgeIndex`, and :obj:`input_value` corresponds to the values
        of non-zero elements in :obj:`input`.
        If not specified, non-zero elements will be assigned a value of
        :obj:`1.0`.

        :obj:`other` can either be a dense :class:`torch.Tensor` or a sparse
        :class:`EdgeIndex`.
        if :obj:`other` is a sparse :class:`EdgeIndex`, then :obj:`other_value`
        corresponds to the values of its non-zero elements.

        This function additionally accepts an optional :obj:`reduce` argument
        that allows specification of an optional reduction operation.
        See :meth:`torch.sparse.mm` for more information.

        Lastly, the :obj:`transpose` option allows to perform matrix
        multiplication where :obj:`input` will be first transposed, *i.e.*:

        .. math::

            \textrm{input}^{\top} \cdot \textrm{other}

        Args:
            other (torch.Tensor or EdgeIndex): The second matrix to be
                multiplied, which can be sparse or dense.
            input_value (torch.Tensor, optional): The values for non-zero
                elements of :obj:`input`.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
            other_value (torch.Tensor, optional): The values for non-zero
                elements of :obj:`other` in case it is sparse.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
            reduce (str, optional): The reduce operation, one of
                :obj:`"sum"`/:obj:`"add"`, :obj:`"mean"`,
                :obj:`"min"`/:obj:`amin` or :obj:`"max"`/:obj:`amax`.
                (default: :obj:`"sum"`)
            transpose (bool, optional): If set to :obj:`True`, will perform
                matrix multiplication based on the transposed :obj:`input`.
                (default: :obj:`False`)
        """
        return matmul(self, other, input_value, other_value, reduce, transpose)

    def sparse_narrow(
        self,
        dim: int,
        start: Union[int, Tensor],
        length: int,
    ) -> 'EdgeIndex':
        r"""Returns a new :class:`EdgeIndex` that is a narrowed version of
        itself. Narrowing is performed by interpreting :class:`EdgeIndex` as a
        sparse matrix of shape :obj:`(num_rows, num_cols)`.

        In contrast to :meth:`torch.narrow`, the returned tensor does not share
        the same underlying storage anymore.

        Args:
            dim (int): The dimension along which to narrow.
            start (int or torch.Tensor): Index of the element to start the
                narrowed dimension from.
            length (int): Length of the narrowed dimension.
        """
        dim = dim + 2 if dim < 0 else dim
        if dim != 0 and dim != 1:
            raise ValueError(f"Expected dimension to be 0 or 1 (got {dim})")

        if start < 0:
            raise ValueError(f"Expected 'start' value to be positive "
                             f"(got {start})")

        if dim == 0:
            if self.is_sorted_by_row:
                (rowptr, col), _ = self.get_csr()
                rowptr = rowptr.narrow(0, start, length + 1)

                if rowptr.numel() < 2:
                    row, col = self._data[0, :0], self._data[1, :0]
                    rowptr = None
                    num_rows = 0
                else:
                    col = col[rowptr[0]:rowptr[-1]]
                    rowptr = rowptr - rowptr[0]
                    num_rows = rowptr.numel() - 1

                    row = torch.arange(
                        num_rows,
                        dtype=col.dtype,
                        device=col.device,
                    ).repeat_interleave(
                        rowptr.diff(),
                        output_size=col.numel(),
                    )

                edge_index = EdgeIndex(
                    torch.stack([row, col], dim=0),
                    sparse_size=(num_rows, self.sparse_size(1)),
                    sort_order='row',
                )
                edge_index._indptr = rowptr
                return edge_index

            else:
                mask = self._data[0] >= start
                mask &= self._data[0] < (start + length)
                offset = torch.tensor([[start], [0]], device=self.device)
                edge_index = self[:, mask].sub_(offset)  # type: ignore
                edge_index._sparse_size = (length, edge_index._sparse_size[1])
                return edge_index

        else:
            assert dim == 1

            if self.is_sorted_by_col:
                (colptr, row), _ = self.get_csc()
                colptr = colptr.narrow(0, start, length + 1)

                if colptr.numel() < 2:
                    row, col = self._data[0, :0], self._data[1, :0]
                    colptr = None
                    num_cols = 0
                else:
                    row = row[colptr[0]:colptr[-1]]
                    colptr = colptr - colptr[0]
                    num_cols = colptr.numel() - 1

                    col = torch.arange(
                        num_cols,
                        dtype=row.dtype,
                        device=row.device,
                    ).repeat_interleave(
                        colptr.diff(),
                        output_size=row.numel(),
                    )

                edge_index = EdgeIndex(
                    torch.stack([row, col], dim=0),
                    sparse_size=(self.sparse_size(0), num_cols),
                    sort_order='col',
                )
                edge_index._indptr = colptr
                return edge_index

            else:
                mask = self._data[1] >= start
                mask &= self._data[1] < (start + length)
                offset = torch.tensor([[0], [start]], device=self.device)
                edge_index = self[:, mask].sub_(offset)  # type: ignore
                edge_index._sparse_size = (edge_index._sparse_size[0], length)
                return edge_index

    def to_vector(self) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a one-dimensional index
        vector representation.
        """
        num_rows, num_cols = self.get_sparse_size()

        if num_rows * num_cols > torch_geometric.typing.MAX_INT64:
            raise ValueError("'to_vector()' will result in an overflow")

        return self._data[0] * num_rows + self._data[1]

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[Any, ...]]:
        attrs = ['_data']
        if self._indptr is not None:
            attrs.append('_indptr')
        if self._T_perm is not None:
            attrs.append('_T_perm')
        # TODO We cannot save `_T_index` for now since it is stored as tuple.
        if self._T_indptr is not None:
            attrs.append('_T_indptr')

        ctx = (
            self._sparse_size,
            self._sort_order,
            self._is_undirected,
            self._cat_metadata,
        )

        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, Any],
        ctx: Tuple[Any, ...],
        outer_size: Tuple[int, ...],
        outer_stride: Tuple[int, ...],
    ) -> 'EdgeIndex':
        edge_index = EdgeIndex(
            inner_tensors['_data'],
            sparse_size=ctx[0],
            sort_order=ctx[1],
            is_undirected=ctx[2],
        )

        edge_index._indptr = inner_tensors.get('_indptr', None)
        edge_index._T_perm = inner_tensors.get('_T_perm', None)
        edge_index._T_indptr = inner_tensors.get('_T_indptr', None)
        edge_index._cat_metadata = ctx[3]

        return edge_index

    # Prevent auto-wrapping outputs back into the proper subclass type:
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(
        cls: Type,
        func: Callable[..., Any],
        types: Iterable[Type[Any]],
        args: Iterable[Tuple[Any, ...]] = (),
        kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        # `EdgeIndex` should be treated as a regular PyTorch tensor for all
        # standard PyTorch functionalities. However,
        # * some of its metadata can be transferred to new functions, e.g.,
        #   `torch.cat(dim=1)` can inherit the sparse matrix size, or
        #   `torch.narrow(dim=1)` can inherit cached pointers.
        # * not all operations lead to valid `EdgeIndex` tensors again, e.g.,
        #   `torch.sum()` does not yield a `EdgeIndex` as its output, or
        #   `torch.cat(dim=0) violates the [2, *] shape assumption.

        # To account for this, we hold a number of `HANDLED_FUNCTIONS` that
        # implement specific functions for valid `EdgeIndex` routines.
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        # For all other PyTorch functions, we treat them as vanilla tensors.
        args = pytree.tree_map_only(EdgeIndex, lambda x: x._data, args)
        if kwargs is not None:
            kwargs = pytree.tree_map_only(EdgeIndex, lambda x: x._data, kwargs)
        return func(*args, **(kwargs or {}))

    def __repr__(self) -> str:  # type: ignore
        prefix = f'{self.__class__.__name__}('
        indent = len(prefix)
        tensor_str = torch._tensor_str._tensor_str(self._data, indent)

        suffixes = []
        num_rows, num_cols = self.sparse_size()
        if num_rows is not None or num_cols is not None:
            size_repr = f"({num_rows or '?'}, {num_cols or '?'})"
            suffixes.append(f'sparse_size={size_repr}')
        suffixes.append(f'nnz={self._data.size(1)}')
        if (self.device.type != torch._C._get_default_device()
                or (self.device.type == 'cuda'
                    and torch.cuda.current_device() != self.device.index)
                or (self.device.type == 'mps')):
            suffixes.append(f"device='{self.device}'")
        if self.dtype != torch.int64:
            suffixes.append(f'dtype={self.dtype}')
        if self.is_sorted:
            suffixes.append(f'sort_order={self.sort_order}')
        if self.is_undirected:
            suffixes.append('is_undirected=True')

        return torch._tensor_str._add_suffixes(prefix + tensor_str, suffixes,
                                               indent, force_newline=False)

    # Helpers #################################################################

    def _shallow_copy(self) -> 'EdgeIndex':
        out = EdgeIndex(self._data)
        out._sparse_size = self._sparse_size
        out._sort_order = self._sort_order
        out._is_undirected = self._is_undirected
        out._indptr = self._indptr
        out._T_perm = self._T_perm
        out._T_index = self._T_index
        out._T_indptr = self._T_indptr
        out._value = self._value
        out._cat_metadata = self._cat_metadata
        return out

    def _clear_metadata(self) -> 'EdgeIndex':
        self._sparse_size = (None, None)
        self._sort_order = None
        self._is_undirected = False
        self._indptr = None
        self._T_perm = None
        self._T_index = (None, None)
        self._T_indptr = None
        self._value = None
        self._cat_metadata = None
        return self


class SortReturnType(NamedTuple):
    values: EdgeIndex
    indices: Optional[Tensor]


def apply_(
    tensor: EdgeIndex,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Union[EdgeIndex, Tensor]:

    data = fn(tensor._data, *args, **kwargs)

    if data.dtype not in INDEX_DTYPES:
        return data

    if tensor._data.data_ptr() != data.data_ptr():
        out = EdgeIndex(data)
    else:  # In-place:
        tensor._data = data
        out = tensor

    # Copy metadata:
    out._sparse_size = tensor._sparse_size
    out._sort_order = tensor._sort_order
    out._is_undirected = tensor._is_undirected
    out._cat_metadata = tensor._cat_metadata

    # Convert cache (but do not consider `_value`):
    if tensor._indptr is not None:
        out._indptr = fn(tensor._indptr, *args, **kwargs)

    if tensor._T_perm is not None:
        out._T_perm = fn(tensor._T_perm, *args, **kwargs)

    _T_row, _T_col = tensor._T_index
    if _T_row is not None:
        _T_row = fn(_T_row, *args, **kwargs)
    if _T_col is not None:
        _T_col = fn(_T_col, *args, **kwargs)
    out._T_index = (_T_row, _T_col)

    if tensor._T_indptr is not None:
        out._T_indptr = fn(tensor._T_indptr, *args, **kwargs)

    return out


@implements(aten.clone.default)
def _clone(
    tensor: EdgeIndex,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> EdgeIndex:
    out = apply_(tensor, aten.clone.default, memory_format=memory_format)
    assert isinstance(out, EdgeIndex)
    return out


@implements(aten._to_copy.default)
def _to_copy(
    tensor: EdgeIndex,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Union[EdgeIndex, Tensor]:
    return apply_(
        tensor,
        aten._to_copy.default,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )


@implements(aten.alias.default)
def _alias(tensor: EdgeIndex) -> EdgeIndex:
    return tensor._shallow_copy()


@implements(aten._pin_memory.default)
def _pin_memory(tensor: EdgeIndex) -> EdgeIndex:
    out = apply_(tensor, aten._pin_memory.default)
    assert isinstance(out, EdgeIndex)
    return out


@implements(aten.cat.default)
def _cat(
    tensors: List[Union[EdgeIndex, Tensor]],
    dim: int = 0,
) -> Union[EdgeIndex, Tensor]:

    data_list = pytree.tree_map_only(EdgeIndex, lambda x: x._data, tensors)
    data = aten.cat.default(data_list, dim=dim)

    if dim != 1 and dim != -1:  # No valid `EdgeIndex` anymore.
        return data

    if any([not isinstance(tensor, EdgeIndex) for tensor in tensors]):
        return data

    out = EdgeIndex(data)

    nnz_list = [t.size(1) for t in tensors]
    sparse_size_list = [t.sparse_size() for t in tensors]  # type: ignore
    sort_order_list = [t._sort_order for t in tensors]  # type: ignore
    is_undirected_list = [t.is_undirected for t in tensors]  # type: ignore

    # Post-process `sparse_size`:
    total_num_rows: Optional[int] = 0
    for num_rows, _ in sparse_size_list:
        if num_rows is None:
            total_num_rows = None
            break
        assert isinstance(total_num_rows, int)
        total_num_rows = max(num_rows, total_num_rows)

    total_num_cols: Optional[int] = 0
    for _, num_cols in sparse_size_list:
        if num_cols is None:
            total_num_cols = None
            break
        assert isinstance(total_num_cols, int)
        total_num_cols = max(num_cols, total_num_cols)

    out._sparse_size = (total_num_rows, total_num_cols)

    # Post-process `is_undirected`:
    out._is_undirected = all(is_undirected_list)

    out._cat_metadata = CatMetadata(
        nnz=nnz_list,
        sparse_size=sparse_size_list,
        sort_order=sort_order_list,
        is_undirected=is_undirected_list,
    )

    return out


@implements(aten.flip.default)
def _flip(
    input: EdgeIndex,
    dims: Union[List[int], Tuple[int, ...]],
) -> EdgeIndex:

    data = aten.flip.default(input._data, dims)
    out = EdgeIndex(data)

    out._value = input._value
    out._is_undirected = input.is_undirected

    # Flip metadata and cache:
    if 0 in dims or -2 in dims:
        out._sparse_size = input.sparse_size()[::-1]

    if len(dims) == 1 and (dims[0] == 0 or dims[0] == -2):
        if input.is_sorted_by_row:
            out._sort_order = SortOrder.COL
        elif input.is_sorted_by_col:
            out._sort_order = SortOrder.ROW

        out._indptr = input._T_indptr
        out._T_perm = input._T_perm
        out._T_index = input._T_index[::-1]
        out._T_indptr = input._indptr

    return out


@implements(aten.index_select.default)
def _index_select(
    input: EdgeIndex,
    dim: int,
    index: Tensor,
) -> Union[EdgeIndex, Tensor]:

    out = aten.index_select.default(input._data, dim, index)

    if dim == 1 or dim == -1:
        out = EdgeIndex(out)
        out._sparse_size = input.sparse_size()

    return out


@implements(aten.slice.Tensor)
def _slice(
    input: EdgeIndex,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> Union[EdgeIndex, Tensor]:

    if ((start is None or start <= 0)
            and (end is None or end > input.size(dim)) and step == 1):
        return input._shallow_copy()  # No-op.

    out = aten.slice.Tensor(input._data, dim, start, end, step)

    if dim == 1 or dim == -1:
        if step != 1:
            out = out.contiguous()

        out = EdgeIndex(out)
        out._sparse_size = input.sparse_size()
        # NOTE We could potentially maintain `rowptr`/`colptr` attributes here,
        # but it is not really clear if this is worth it. The most important
        # information, the sort order, needs to be maintained though:
        if step >= 0:
            out._sort_order = input._sort_order
        else:
            if input._sort_order == SortOrder.ROW:
                out._sort_order = SortOrder.COL
            elif input._sort_order == SortOrder.COL:
                out._sort_order = SortOrder.ROW

    return out


@implements(aten.index.Tensor)
def _index(
    input: Union[EdgeIndex, Tensor],
    indices: List[Optional[Union[Tensor, EdgeIndex]]],
) -> Union[EdgeIndex, Tensor]:

    if not isinstance(input, EdgeIndex):
        indices = pytree.tree_map_only(EdgeIndex, lambda x: x._data, indices)
        return aten.index.Tensor(input, indices)

    out = aten.index.Tensor(input._data, indices)

    if len(indices) != 2 or indices[0] is not None:
        return out

    index = indices[1]
    assert isinstance(index, Tensor)

    out = EdgeIndex(out)

    # 1. `edge_index[:, mask]` or `edge_index[..., mask]`.
    if index.dtype in (torch.bool, torch.uint8):
        out._sparse_size = input.sparse_size()
        out._sort_order = input._sort_order

    else:  # 2. `edge_index[:, index]` or `edge_index[..., index]`.
        out._sparse_size = input.sparse_size()

    return out


@implements(aten.select.int)
def _select(input: EdgeIndex, dim: int, index: int) -> Union[Tensor, Index]:
    out = aten.select.int(input._data, dim, index)

    if dim == 0 or dim == -2:
        out = Index(out)

        if index == 0 or index == -2:  # Row-select:
            out._dim_size = input.sparse_size(0)
            out._is_sorted = input.is_sorted_by_row
            if input.is_sorted_by_row:
                out._indptr = input._indptr

        else:  # Col-select:
            assert index == 1 or index == -1
            out._dim_size = input.sparse_size(1)
            out._is_sorted = input.is_sorted_by_col
            if input.is_sorted_by_col:
                out._indptr = input._indptr

    return out


@implements(aten.unbind.int)
def _unbind(
    input: EdgeIndex,
    dim: int = 0,
) -> Union[List[Index], List[Tensor]]:

    if dim == 0 or dim == -2:
        row = input[0]
        assert isinstance(row, Index)
        col = input[1]
        assert isinstance(col, Index)
        return [row, col]

    return aten.unbind.int(input._data, dim)


@implements(aten.add.Tensor)
def _add(
    input: EdgeIndex,
    other: Union[int, Tensor, EdgeIndex],
    *,
    alpha: int = 1,
) -> Union[EdgeIndex, Tensor]:

    out = aten.add.Tensor(
        input._data,
        other._data if isinstance(other, EdgeIndex) else other,
        alpha=alpha,
    )

    if out.dtype not in INDEX_DTYPES:
        return out
    if out.dim() != 2 or out.size(0) != 2:
        return out

    out = EdgeIndex(out)

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        size = maybe_add(input._sparse_size, other, alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = input._sort_order
        out._is_undirected = input.is_undirected
        out._T_perm = input._T_perm

    elif isinstance(other, Tensor) and other.size() == (2, 1):
        size = maybe_add(input._sparse_size, other.view(-1).tolist(), alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = input._sort_order
        if torch.equal(other[0], other[1]):
            out._is_undirected = input.is_undirected
        out._T_perm = input._T_perm

    elif isinstance(other, EdgeIndex):
        size = maybe_add(input._sparse_size, other._sparse_size, alpha)
        assert len(size) == 2
        out._sparse_size = size

    return out


@implements(aten.add_.Tensor)
def add_(
    input: EdgeIndex,
    other: Union[int, Tensor, EdgeIndex],
    *,
    alpha: int = 1,
) -> EdgeIndex:

    sparse_size = input._sparse_size
    sort_order = input._sort_order
    is_undirected = input._is_undirected
    T_perm = input._T_perm
    input._clear_metadata()

    aten.add_.Tensor(
        input._data,
        other._data if isinstance(other, EdgeIndex) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        size = maybe_add(sparse_size, other, alpha)
        assert len(size) == 2
        input._sparse_size = size
        input._sort_order = sort_order
        input._is_undirected = is_undirected
        input._T_perm = T_perm

    elif isinstance(other, Tensor) and other.size() == (2, 1):
        size = maybe_add(sparse_size, other.view(-1).tolist(), alpha)
        assert len(size) == 2
        input._sparse_size = size
        input._sort_order = sort_order
        if torch.equal(other[0], other[1]):
            input._is_undirected = is_undirected
        input._T_perm = T_perm

    elif isinstance(other, EdgeIndex):
        size = maybe_add(sparse_size, other._sparse_size, alpha)
        assert len(size) == 2
        input._sparse_size = size

    return input


@implements(aten.sub.Tensor)
def _sub(
    input: EdgeIndex,
    other: Union[int, Tensor, EdgeIndex],
    *,
    alpha: int = 1,
) -> Union[EdgeIndex, Tensor]:

    out = aten.sub.Tensor(
        input._data,
        other._data if isinstance(other, EdgeIndex) else other,
        alpha=alpha,
    )

    if out.dtype not in INDEX_DTYPES:
        return out
    if out.dim() != 2 or out.size(0) != 2:
        return out

    out = EdgeIndex(out)

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        size = maybe_sub(input._sparse_size, other, alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = input._sort_order
        out._is_undirected = input.is_undirected
        out._T_perm = input._T_perm

    elif isinstance(other, Tensor) and other.size() == (2, 1):
        size = maybe_sub(input._sparse_size, other.view(-1).tolist(), alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = input._sort_order
        if torch.equal(other[0], other[1]):
            out._is_undirected = input.is_undirected
        out._T_perm = input._T_perm

    return out


@implements(aten.sub_.Tensor)
def sub_(
    input: EdgeIndex,
    other: Union[int, Tensor, EdgeIndex],
    *,
    alpha: int = 1,
) -> EdgeIndex:

    sparse_size = input._sparse_size
    sort_order = input._sort_order
    is_undirected = input._is_undirected
    T_perm = input._T_perm
    input._clear_metadata()

    aten.sub_.Tensor(
        input._data,
        other._data if isinstance(other, EdgeIndex) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        size = maybe_sub(sparse_size, other, alpha)
        assert len(size) == 2
        input._sparse_size = size
        input._sort_order = sort_order
        input._is_undirected = is_undirected
        input._T_perm = T_perm

    elif isinstance(other, Tensor) and other.size() == (2, 1):
        size = maybe_sub(sparse_size, other.view(-1).tolist(), alpha)
        assert len(size) == 2
        input._sparse_size = size
        input._sort_order = sort_order
        if torch.equal(other[0], other[1]):
            input._is_undirected = is_undirected
        input._T_perm = T_perm

    return input


# Sparse-Dense Matrix Multiplication ##########################################


def _torch_sparse_spmm(
    input: EdgeIndex,
    other: Tensor,
    value: Optional[Tensor] = None,
    reduce: ReduceType = 'sum',
    transpose: bool = False,
) -> Tensor:
    # `torch-sparse` still provides a faster sparse-dense matrix multiplication
    # code path on GPUs (after all these years...):
    assert torch_geometric.typing.WITH_TORCH_SPARSE
    reduce = PYG_REDUCE[reduce] if reduce in PYG_REDUCE else reduce

    # Optional arguments for backpropagation:
    colptr: Optional[Tensor] = None
    perm: Optional[Tensor] = None

    if not transpose:
        assert input.is_sorted_by_row
        (rowptr, col), _ = input.get_csr()
        row = input._data[0]
        if other.requires_grad and reduce in ['sum', 'mean']:
            (colptr, _), perm = input.get_csc()
    else:
        assert input.is_sorted_by_col
        (rowptr, col), _ = input.get_csc()
        row = input._data[1]
        if other.requires_grad and reduce in ['sum', 'mean']:
            (colptr, _), perm = input.get_csr()

    if reduce == 'sum':
        return torch.ops.torch_sparse.spmm_sum(  #
            row, rowptr, col, value, colptr, perm, other)

    if reduce == 'mean':
        rowcount = rowptr.diff() if other.requires_grad else None
        return torch.ops.torch_sparse.spmm_mean(  #
            row, rowptr, col, value, rowcount, colptr, perm, other)

    if reduce == 'min':
        return torch.ops.torch_sparse.spmm_min(rowptr, col, value, other)[0]

    if reduce == 'max':
        return torch.ops.torch_sparse.spmm_max(rowptr, col, value, other)[0]

    raise NotImplementedError


class _TorchSPMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: EdgeIndex,
        other: Tensor,
        value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
        transpose: bool = False,
    ) -> Tensor:

        reduce = TORCH_REDUCE[reduce] if reduce in TORCH_REDUCE else reduce

        value = value.detach() if value is not None else value
        if other.requires_grad:
            other = other.detach()
            ctx.save_for_backward(input, value)
            ctx.reduce = reduce
            ctx.transpose = transpose

        if not transpose:
            assert input.is_sorted_by_row
            adj = input.to_sparse_csr(value)
        else:
            assert input.is_sorted_by_col
            adj = input.to_sparse_csc(value).t()

        if torch_geometric.typing.WITH_PT20 and not other.is_cuda:
            return torch.sparse.mm(adj, other, reduce)
        else:  # pragma: no cover
            assert reduce == 'sum'
            return adj @ other

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Tuple[None, Optional[Tensor], None, None, None]:

        grad_out, = grad_outputs

        other_grad: Optional[Tensor] = None
        if ctx.needs_input_grad[1]:
            input, value = ctx.saved_tensors
            assert ctx.reduce == 'sum'

            if not ctx.transpose:
                if value is None and input.is_undirected:
                    adj = input.to_sparse_csr(value)
                else:
                    (colptr, row), perm = input.get_csc()
                    if value is not None and perm is not None:
                        value = value[perm]
                    else:
                        value = input._get_value()
                    adj = torch.sparse_csr_tensor(
                        crow_indices=colptr,
                        col_indices=row,
                        values=value,
                        size=input.get_sparse_size()[::-1],
                        device=input.device,
                    )
            else:
                if value is None and input.is_undirected:
                    adj = input.to_sparse_csc(value).t()
                else:
                    (rowptr, col), perm = input.get_csr()
                    if value is not None and perm is not None:
                        value = value[perm]
                    else:
                        value = input._get_value()
                    adj = torch.sparse_csr_tensor(
                        crow_indices=rowptr,
                        col_indices=col,
                        values=value,
                        size=input.get_sparse_size()[::-1],
                        device=input.device,
                    )

            other_grad = adj @ grad_out

        if ctx.needs_input_grad[2]:
            raise NotImplementedError("Gradient computation for 'value' not "
                                      "yet supported")

        return None, other_grad, None, None, None


def _scatter_spmm(
    input: EdgeIndex,
    other: Tensor,
    value: Optional[Tensor] = None,
    reduce: ReduceType = 'sum',
    transpose: bool = False,
) -> Tensor:
    from torch_geometric.utils import scatter

    if not transpose:
        other_j = other[input._data[1]]
        index = input._data[0]
        dim_size = input.get_sparse_size(0)
    else:
        other_j = other[input._data[0]]
        index = input._data[1]
        dim_size = input.get_sparse_size(1)

    other_j = other_j * value.view(-1, 1) if value is not None else other_j
    return scatter(other_j, index, 0, dim_size=dim_size, reduce=reduce)


def _spmm(
    input: EdgeIndex,
    other: Tensor,
    value: Optional[Tensor] = None,
    reduce: ReduceType = 'sum',
    transpose: bool = False,
) -> Tensor:

    if reduce not in get_args(ReduceType):
        raise ValueError(f"`reduce='{reduce}'` is not a valid reduction")

    if not transpose and not input.is_sorted_by_row:
        cls_name = input.__class__.__name__
        raise ValueError(f"'matmul(..., transpose=False)' requires "
                         f"'{cls_name}' to be sorted by rows")

    if transpose and not input.is_sorted_by_col:
        cls_name = input.__class__.__name__
        raise ValueError(f"'matmul(..., transpose=True)' requires "
                         f"'{cls_name}' to be sorted by columns")

    if (torch_geometric.typing.WITH_TORCH_SPARSE and not is_compiling()
            and other.is_cuda):  # pragma: no cover
        return _torch_sparse_spmm(input, other, value, reduce, transpose)

    if value is not None and value.requires_grad:
        if torch_geometric.typing.WITH_TORCH_SPARSE and not is_compiling():
            return _torch_sparse_spmm(input, other, value, reduce, transpose)
        return _scatter_spmm(input, other, value, reduce, transpose)

    if torch_geometric.typing.WITH_PT20:
        if reduce == 'sum' or reduce == 'add':
            return _TorchSPMM.apply(input, other, value, 'sum', transpose)

        if reduce == 'mean':
            out = _TorchSPMM.apply(input, other, value, 'sum', transpose)
            count = input.get_indptr().diff()
            return out / count.clamp_(min=1).to(out.dtype).view(-1, 1)

        if not other.is_cuda and not other.requires_grad:
            return _TorchSPMM.apply(input, other, value, reduce, transpose)

    if torch_geometric.typing.WITH_TORCH_SPARSE and not is_compiling():
        return _torch_sparse_spmm(input, other, value, reduce, transpose)

    return _scatter_spmm(input, other, value, reduce, transpose)


def matmul(
    input: EdgeIndex,
    other: Union[Tensor, EdgeIndex],
    input_value: Optional[Tensor] = None,
    other_value: Optional[Tensor] = None,
    reduce: ReduceType = 'sum',
    transpose: bool = False,
) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:

    if not isinstance(other, EdgeIndex):
        if other_value is not None:
            raise ValueError("'other_value' not supported for sparse-dense "
                             "matrix multiplication")
        return _spmm(input, other, input_value, reduce, transpose)

    if reduce not in ['sum', 'add']:
        raise NotImplementedError(f"`reduce='{reduce}'` not yet supported for "
                                  f"sparse-sparse matrix multiplication")

    transpose &= not input.is_undirected or input_value is not None

    if torch_geometric.typing.NO_MKL:  # pragma: no cover
        sparse_input = input.to_sparse_coo(input_value)
    elif input.is_sorted_by_col:
        sparse_input = input.to_sparse_csc(input_value)
    else:
        sparse_input = input.to_sparse_csr(input_value)

    if transpose:
        sparse_input = sparse_input.t()

    if torch_geometric.typing.NO_MKL:  # pragma: no cover
        other = other.to_sparse_coo(other_value)
    elif other.is_sorted_by_col:
        other = other.to_sparse_csc(other_value)
    else:
        other = other.to_sparse_csr(other_value)

    out = torch.matmul(sparse_input, other)

    rowptr: Optional[Tensor] = None
    if out.layout == torch.sparse_csr:
        rowptr = out.crow_indices().to(input.dtype)
        col = out.col_indices().to(input.dtype)
        edge_index = torch._convert_indices_from_csr_to_coo(
            rowptr, col, out_int32=rowptr.dtype != torch.int64)

    elif out.layout == torch.sparse_coo:  # pragma: no cover
        out = out.coalesce()
        edge_index = out.indices()

    else:
        raise NotImplementedError

    edge_index = EdgeIndex(edge_index)
    edge_index._sort_order = SortOrder.ROW
    edge_index._sparse_size = (out.size(0), out.size(1))
    edge_index._indptr = rowptr

    return edge_index, out.values()


@implements(aten.mm.default)
def _mm(
    input: EdgeIndex,
    other: Union[Tensor, EdgeIndex],
) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:
    return matmul(input, other)


@implements(aten._sparse_addmm.default)
def _addmm(
    input: Tensor,
    mat1: EdgeIndex,
    mat2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    assert input.abs().sum() == 0.0
    out = matmul(mat1, mat2)
    assert isinstance(out, Tensor)
    return alpha * out if alpha != 1.0 else out


if hasattr(aten, '_sparse_mm_reduce_impl'):

    @implements(aten._sparse_mm_reduce_impl.default)
    def _mm_reduce(
        mat1: EdgeIndex,
        mat2: Tensor,
        reduce: ReduceType = 'sum',
    ) -> Tuple[Tensor, Tensor]:
        out = matmul(mat1, mat2, reduce=reduce)
        assert isinstance(out, Tensor)
        return out, out  # We return a dummy tensor for `argout` for now.
