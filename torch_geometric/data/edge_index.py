import functools
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import index_sort

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}

SUPPORTED_DTYPES: Set[torch.dtype] = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

ReduceType = Literal['sum']


class SortOrder(Enum):
    ROW = 'row'
    COL = 'col'


def implements(torch_function: Callable):
    r"""Registers a :pytorch:`PyTorch` function override."""
    @functools.wraps(torch_function)
    def decorator(my_function: Callable):
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


def assert_valid_dtype(tensor: Tensor):
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"'EdgeIndex' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{SUPPORTED_DTYPES})")


def assert_two_dimensional(tensor: Tensor):
    if tensor.dim() != 2:
        raise ValueError(f"'EdgeIndex' needs to be two-dimensional "
                         f"(got {tensor.dim()} dimensions)")
    if tensor.size(0) != 2:
        raise ValueError(f"'EdgeIndex' needs to have a shape of "
                         f"[2, *] (got {list(tensor.size())})")


def assert_contiguous(tensor: Tensor):
    if not tensor.is_contiguous():
        raise ValueError("'EdgeIndex' needs to be contiguous. Please call "
                         "`edge_index.contiguous()` before proceeding.")


class EdgeIndex(Tensor):
    r"""An advanced :obj:`edge_index` representation with additional (meta)data
    attached.

    :class:`EdgeIndex` is a :pytorch:`PyTorch` tensor, that holds an
    :obj:`edge_index` representation of shape :obj:`[2, num_edges]`.
    Edges are given as pairwise source and destination node indices in sparse
    COO format.

    While :class:`EdgeIndex` sub-classes a general :pytorch:`PyTorch` tensor,
    it can hold additional (meta)data, *i.e.*:

    * :obj:`sparse_size`: The underlying sparse matrix size
    * :obj:`sort_order`: The sort order (if present), either by row or column.

    Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
    in case its representation is sorted, such as its :obj:`rowptr` or
    :obj:`colptr`, or the permutation vectors for fast conversion from CSR to
    CSC and vice versa.
    Caches are filled based on demand (*e.g.*, when calling
    :meth:`EdgeIndex.sort_by`), or when explicitly requested via
    :meth:`EdgeIndex.fill_cache`, and are maintained and adjusted over its
    lifespan (*e.g.*, when calling :meth:`EdgeIndex.flip`).

    This representation ensures for optimal computation in GNN message passing
    schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
    workflows.
    """
    # See "https://pytorch.org/docs/stable/notes/extending.html"
    # for a basic tutorial on how to subclass `torch.Tensor`.

    # The size of the underlying sparse matrix:
    _sparse_size: Tuple[Optional[int], Optional[int]] = (None, None)

    # Whether the `edge_index` represented is non-sorted (`None`), or sorted
    # based on row or column values.
    _sort_order: Optional[SortOrder] = None

    # An additional data cache:
    _rowptr: Optional[Tensor] = None  # The CSR `rowptr` in case sorted by row.
    _colptr: Optional[Tensor] = None  # The CSC `colptr` in case sorted by col.
    _csr2csc: Optional[Tensor] = None  # Permutation from CSR to CSC.
    _csc2csr: Optional[Tensor] = None  # Permutation from CSC to CSR.

    _value: Optional[Tensor] = None  # 1-element value for SpMM.

    def __new__(
        cls,
        data,
        *args,
        sparse_size: Tuple[Optional[int], Optional[int]] = (None, None),
        sort_order: Optional[Union[str, SortOrder]] = None,
        **kwargs,
    ):
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
        assert_valid_dtype(data)
        assert_two_dimensional(data)
        assert_contiguous(data)

        out = super().__new__(cls, data)

        # Attach metadata:
        out._sparse_size = sparse_size
        out._sort_order = None if sort_order is None else SortOrder(sort_order)

        return out

    def validate(self) -> 'EdgeIndex':
        r"""Validates the :class:`EdgeIndex` representation, i.e., it ensures
        * that :class:`EdgeIndex` only holds valid entries.
        * that the sort order is correctly set.
        """
        assert_valid_dtype(self)
        assert_two_dimensional(self)
        assert_contiguous(self)

        if self.numel() > 0 and self.min() < 0:
            raise ValueError(f"'{self.__class__.__name__}' contains negative "
                             f"indices (got {int(self.min())})")

        if (self.numel() > 0 and self.num_rows is not None
                and self[0].max() >= self.num_rows):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its number of rows "
                             f"(got {int(self[0].max())}, but expected values "
                             f"smaller than {self.num_rows})")

        if (self.numel() > 0 and self.num_cols is not None
                and self[1].max() >= self.num_cols):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its number of columns "
                             f"(got {int(self[1].max())}, but expected values "
                             f"smaller than {self.num_cols})")

        if self._sort_order == SortOrder.ROW and (self[0].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"row indices")

        if self._sort_order == SortOrder.COL and (self[1].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"column indices")

        return self

    @property
    def sparse_size(self) -> Tuple[Optional[int], Optional[int]]:
        r"""The size of the underlying sparse matrix."""
        return self._sparse_size

    def get_sparse_size(self) -> Tuple[int, int]:
        r"""The size of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        return (self.get_num_rows(), self.get_num_cols())

    @property
    def num_rows(self) -> Optional[int]:
        r"""The number of rows of the underlying sparse matrix."""
        return self._sparse_size[0]

    def get_num_rows(self) -> int:
        r"""The number of rows of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        if self.num_rows is None:
            self._sparse_size = (
                int(self[0].max()) + 1 if self.numel() > 0 else 0,
                self.num_cols,
            )

        return self.num_rows

    @property
    def num_cols(self) -> Optional[int]:
        r"""The number of columns of the underlying sparse matrix."""
        return self._sparse_size[1]

    def get_num_cols(self) -> int:
        r"""The number of columns of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        if self.num_cols is None:
            self._sparse_size = (
                self.num_rows,
                int(self[1].max()) + 1 if self.numel() > 0 else 0,
            )

        return self.num_cols

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

    def get_rowptr(self) -> Tensor:
        r"""Returns the :obj:`rowptr` vector of :class:`EdgeIndex`, a
        compressed representation of row indices in case :class:`EdgeIndex` is
        sorted by rows.
        """
        if self._sort_order != SortOrder.ROW:
            raise ValueError(
                f"Cannot access 'rowptr' in '{self.__class__.__name__}' "
                f"since it is not sorted by rows (got '{self.sort_order}')")

        if self._rowptr is not None:
            return self._rowptr

        self._rowptr = torch._convert_indices_from_coo_to_csr(
            self[0], self.get_num_rows(), out_int32=self.dtype != torch.int64)
        self._rowptr = self._rowptr.to(self.dtype)

        return self._rowptr

    def get_colptr(self) -> Tensor:
        r"""Returns the :obj:`colptr` vector of :class:`EdgeIndex`, a
        compressed representation of column indices in case :class:`EdgeIndex`
        is sorted by columns.
        """
        if self._sort_order != SortOrder.COL:
            raise ValueError(
                f"Cannot access 'colptr' in '{self.__class__.__name__}' "
                f"since it is not sorted by columns (got '{self.sort_order}')")

        if self._colptr is not None:
            return self._colptr

        self._colptr = torch._convert_indices_from_coo_to_csr(
            self[1], self.get_num_cols(), out_int32=self.dtype != torch.int64)
        self._colptr = self._colptr.to(self.dtype)

        return self._colptr

    def _get_value(self, dtype: Optional[torch.dtype] = None) -> Tensor:
        if self._value is not None:
            if (dtype or torch.get_default_dtype()) == self._value.dtype:
                return self._value

        if (torch_geometric.typing.WITH_PT20
                and not torch_geometric.typing.WITH_ARM):
            value = torch.ones(1, dtype=dtype, device=self.device)
            value = value.expand(self.size(1))
        else:
            value = torch.ones(self.size(1), dtype=dtype, device=self.device)

        self._value = value

        return self._value

    def fill_cache_(self) -> 'EdgeIndex':
        r"""Fills the cache with (meta)data information.
        No-op in case :class:`EdgeIndex` is not sorted.
        """
        self.get_num_rows()
        self.get_num_cols()

        if self._sort_order == SortOrder.ROW:
            self.get_rowptr()
        if self._sort_order == SortOrder.COL:
            self.get_colptr()

        return self

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`EdgeIndex` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self.as_subclass(Tensor)

    def sort_by(
        self,
        sort_order: Union[str, SortOrder],
        stable: bool = False,
    ) -> torch.return_types.sort:
        r"""Sorts the elements by row or column values.

        Args:
            sort_order (str): The sort order, either :obj:`"row"` or
                :obj:`"col"`.
            stable (bool, optional): Makes the sorting routine stable, which
                guarantees that the order of equivalent elements is preserved.
                (default: :obj:`False`)
        """
        sort_order = SortOrder(sort_order)

        if self._sort_order == sort_order:  # Nothing to do.
            perm = torch.arange(self.size(1), device=self.device)
            return torch.return_types.sort([self, perm])

        # If conversion from CSR->CSC or CSC->CSR is known, make use of it:
        if self._sort_order == SortOrder.ROW and self._csr2csc is not None:
            edge_index = self.as_tensor()[:, self._csr2csc]
            perm = self._csr2csc

        elif self._sort_order == SortOrder.COL and self._csc2csr is not None:
            edge_index = self.as_tensor()[:, self._csc2csr]
            perm = self._csc2csr

        # Otherwise, perform sorting:
        elif sort_order == SortOrder.ROW:
            row, perm = index_sort(self.as_tensor()[0], self.num_rows, stable)
            edge_index = torch.stack([row, self.as_tensor()[1][perm]], dim=0)

        else:
            col, perm = index_sort(self.as_tensor()[1], self.num_cols, stable)
            edge_index = torch.stack([self.as_tensor()[0][perm], col], dim=0)

        out = self.__class__(edge_index)

        # We can fully fill metadata and cache:
        out._sparse_size = self.sparse_size
        out._sort_order = sort_order
        out._rowptr = self._rowptr
        out._colptr = self._colptr
        out._csr2csc = self._csr2csc
        out._csc2csr = self._csc2csr
        out._value = self._value

        # Fill information for faster future CSR->CSC or CSC->CSR conversion:
        if self._sort_order == SortOrder.ROW:
            out._csr2csc = self._csr2csc = perm
        elif self._sort_order == SortOrder.COL:
            out._csc2csr = self._csc2csr = perm

        return torch.return_types.sort([out, perm])

    def to_sparse_tensor(
        self,
        value: Optional[Tensor] = None,
    ) -> SparseTensor:
        r"""Converts the :class:`EdgeIndex` representation to a
        :class:`torch_sparse.SparseTensor`. Requires that :obj:`torch-sparse`
        is installed.

        Args:
            value (torch.Tensor, optional): The values of non-zero indices.
                (default: :obj:`None`)
        """
        return SparseTensor(
            row=self[0],
            col=self[1],
            rowptr=self.get_rowptr() if self.is_sorted_by_row else None,
            value=value,
            sparse_sizes=self.get_sparse_size(),
            is_sorted=self.is_sorted_by_row,
            trust_data=True,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
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

        # For all other PyTorch functions, we return a vanilla PyTorch tensor.
        types = tuple(torch.Tensor if issubclass(t, cls) else t for t in types)
        return Tensor.__torch_function__(func, types, args, kwargs)


def apply_(
    tensor: EdgeIndex,
    fn: Callable,
    *args,
    **kwargs,
) -> EdgeIndex:

    out = Tensor.__torch_function__(fn, (Tensor, ), (tensor, ) + args, kwargs)
    out = out.as_subclass(EdgeIndex)

    # Copy metadata:
    out._sparse_size = tensor.sparse_size
    out._sort_order = tensor._sort_order

    # Convert cache (but do not consider `_value`):
    if tensor._rowptr is not None:
        out._rowptr = fn(tensor._rowptr, *args, **kwargs)
    if tensor._colptr is not None:
        out._colptr = fn(tensor._colptr, *args, **kwargs)
    if tensor._csr2csc is not None:
        out._csr2csc = fn(tensor._csr2csc, *args, **kwargs)
    if tensor._csc2csr is not None:
        out._csc2csr = fn(tensor._csc2csr, *args, **kwargs)

    return out


@implements(torch.clone)
@implements(Tensor.clone)
def clone(tensor: EdgeIndex) -> EdgeIndex:
    return apply_(tensor, Tensor.clone)


@implements(Tensor.to)
def to(tensor: EdgeIndex, *args, **kwargs) -> EdgeIndex:
    out = apply_(tensor, Tensor.to, *args, **kwargs)

    if out.dtype not in SUPPORTED_DTYPES:
        out = out.as_tensor()

    return out


@implements(Tensor.cpu)
def cpu(tensor: EdgeIndex, *args, **kwargs) -> EdgeIndex:
    return apply_(tensor, Tensor.cpu, *args, **kwargs)


@implements(Tensor.cuda)
def cuda(tensor: EdgeIndex, *args, **kwargs) -> EdgeIndex:
    return apply_(tensor, Tensor.cuda, *args, **kwargs)


@implements(Tensor.share_memory_)
def share_memory_(tensor: EdgeIndex) -> EdgeIndex:
    return apply_(tensor, Tensor.share_memory_)


@implements(Tensor.contiguous)
def contiguous(tensor: EdgeIndex) -> EdgeIndex:
    return apply_(tensor, Tensor.contiguous)


@implements(torch.cat)
def cat(
    tensors: List[Union[EdgeIndex, Tensor]],
    dim: int = 0,
) -> Union[EdgeIndex, Tensor]:

    out = Tensor.__torch_function__(torch.cat, (Tensor, ), (tensors, dim))

    if dim != 1 and dim != -1:  # No valid `EdgeIndex` anymore.
        return out

    out = out.as_subclass(EdgeIndex)

    # Post-process `sparse_size`:
    num_rows = 0
    for tensor in tensors:
        if not isinstance(tensor, EdgeIndex) or tensor.num_rows is None:
            num_rows = None
            break
        num_rows = max(num_rows, tensor.num_rows)

    num_cols = 0
    for tensor in tensors:
        if not isinstance(tensor, EdgeIndex) or tensor.num_cols is None:
            num_cols = None
            break
        num_cols = max(num_cols, tensor.num_cols)

    out._sparse_size = (num_rows, num_cols)

    return out


@implements(torch.flip)
@implements(Tensor.flip)
def flip(
    input: EdgeIndex,
    dims: Union[int, List[int], Tuple[int, ...]],
) -> Union[EdgeIndex, Tensor]:

    if isinstance(dims, int):
        dims = [dims]
    assert isinstance(dims, (tuple, list))

    out = Tensor.__torch_function__(torch.flip, (Tensor, ), (input, dims))
    out = out.as_subclass(EdgeIndex)

    # Flip metadata and cache:
    if 0 in dims or -2 in dims:
        out._sparse_size = input.sparse_size[::-1]

    if len(dims) == 1 and (dims[0] == 0 or dims[0] == -2):
        if input._sort_order == SortOrder.ROW:
            out._sort_order = SortOrder.COL
        elif input._sort_order == SortOrder.COL:
            out._sort_order = SortOrder.ROW

        out._rowptr = input._colptr
        out._colptr = input._rowptr
        out._csr2csc = input._csc2csr
        out._csc2csr = input._csr2csc

    out._value = input._value

    return out


@implements(torch.index_select)
@implements(Tensor.index_select)
def index_select(
    input: EdgeIndex,
    dim: int,
    index: Tensor,
) -> Union[EdgeIndex, Tensor]:

    out = Tensor.__torch_function__(  #
        torch.index_select, (Tensor, ), (input, dim, index))

    if dim == 1 or dim == -1:
        out = out.as_subclass(EdgeIndex)
        out._sparse_size = input.sparse_size

    return out


@implements(torch.narrow)
@implements(Tensor.narrow)
def narrow(
    input: EdgeIndex,
    dim: int,
    start: Union[int, Tensor],
    length: int,
) -> Union[EdgeIndex, Tensor]:

    out = Tensor.__torch_function__(  #
        torch.narrow, (Tensor, ), (input, dim, start, length))

    if dim == 1 or dim == -1:
        out = out.as_subclass(EdgeIndex)
        out._sparse_size = input.sparse_size
        # NOTE We could potentially maintain `rowptr`/`colptr` attributes here,
        # but it is not really clear if this is worth it. The most important
        # information, the sort order, needs to be maintained though:
        out._sort_order = input._sort_order

    return out


@implements(Tensor.__getitem__)
def getitem(input: EdgeIndex, index: Any) -> Union[EdgeIndex, Tensor]:
    out = Tensor.__torch_function__(  #
        Tensor.__getitem__, (Tensor, ), (input, index))

    # There exists 3 possible index types that map back to a valid `EdgeIndex`,
    # and all include selecting/filtering in the last dimension only:
    def is_last_dim_select(i: Any) -> bool:
        # Maps to true for `__getitem__` requests of the form
        # `tensor[..., index]` or `tensor[:, index]`.
        if not isinstance(i, tuple) or len(i) != 2:
            return False
        if i[0] == Ellipsis:
            return True
        if not isinstance(i[0], slice):
            return False
        return i[0].start is None and i[0].stop is None and i[0].step is None

    is_valid = is_last_dim_select(index)

    # 1. `edge_index[:, mask]` or `edge_index[..., mask]`.
    if is_valid and isinstance(index[1], (torch.BoolTensor, torch.ByteTensor)):
        out = out.as_subclass(EdgeIndex)
        out._sparse_size = input.sparse_size
        out._sort_order = input._sort_order

    # 2. `edge_index[:, index]` or `edge_index[..., index]`.
    elif is_valid and isinstance(index[1], Tensor):
        out = out.as_subclass(EdgeIndex)
        out._sparse_size = input.sparse_size

    # 3. `edge_index[:, slice]` or `edge_index[..., slice]`.
    elif is_valid and isinstance(index[1], slice):
        out = out.as_subclass(EdgeIndex)
        out._sparse_size = input.sparse_size
        if index[1].step is None or index[1].step > 0:
            out._sort_order = input._sort_order

    return out


@implements(Tensor.to_dense)
def to_dense(
    tensor: EdgeIndex,
    dtype: Optional[torch.dtype] = None,
    value: Optional[Tensor] = None,
) -> Tensor:

    # TODO Respect duplicate edges.

    dtype = value.dtype if value is not None else dtype

    size = tensor.get_sparse_size()
    if value is not None and value.dim() > 1:
        size = size + value.shape[1:]

    out = torch.zeros(size, dtype=dtype, device=tensor.device)
    out[tensor[0], tensor[1]] = value if value is not None else 1

    return out


# `to_sparse_coo()` uses `to_sparse(layout=None)` dispatch logic:
def to_sparse_coo(tensor: EdgeIndex, value: Optional[Tensor] = None) -> Tensor:
    out = torch.sparse_coo_tensor(
        indices=tensor.as_tensor(),
        values=tensor._get_value() if value is None else value,
        size=tensor.get_sparse_size(),
        device=tensor.device,
    )

    if tensor._sort_order == SortOrder.ROW:
        out = out._coalesced_(True)

    return out


@implements(Tensor.to_sparse_csr)
def to_sparse_csr(tensor: EdgeIndex, value: Optional[Tensor] = None) -> Tensor:
    return torch.sparse_csr_tensor(
        crow_indices=tensor.get_rowptr(),
        col_indices=tensor[1],
        values=tensor._get_value() if value is None else value,
        size=tensor.get_sparse_size(),
        device=tensor.device,
    )


if torch_geometric.typing.WITH_PT112:

    @implements(Tensor.to_sparse_csc)
    def to_sparse_csc(
        tensor: EdgeIndex,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        return torch.sparse_csc_tensor(
            ccol_indices=tensor.get_colptr(),
            row_indices=tensor[0],
            values=tensor._get_value() if value is None else value,
            size=tensor.get_sparse_size(),
            device=tensor.device,
        )

else:

    def to_sparse_csc(
        tensor: EdgeIndex,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError(
            "'to_sparse_csc' not supported for PyTorch < 1.12")


if torch_geometric.typing.WITH_PT20:

    @implements(Tensor.to_sparse)
    def to_sparse(
        tensor: EdgeIndex,
        *,
        layout: Optional[torch.layout] = None,
        value: Optional[Tensor] = None,
    ) -> Tensor:

        if layout is None or layout == torch.sparse_coo:
            return to_sparse_coo(tensor, value)
        if layout == torch.sparse_csr:
            return to_sparse_csr(tensor, value)
        if torch_geometric.typing.WITH_PT112 and layout == torch.sparse_csc:
            return to_sparse_csc(tensor, value)

        raise ValueError(f"Unexpected tensor layout (got '{layout}')")

else:

    @implements(Tensor.to_sparse)
    def to_sparse(tensor: EdgeIndex, value: Optional[Tensor] = None) -> Tensor:
        return to_sparse_coo(tensor, value)


class SparseDenseMatmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: EdgeIndex,
        other: Tensor,
        input_value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
    ) -> Tensor:

        if input._sort_order != SortOrder.ROW:
            raise ValueError("Sparse-dense matrix multiplication requires "
                             "'EdgeIndex' to be sorted by rows")

        if reduce not in ReduceType.__args__:
            raise NotImplementedError(f"`reduce='{reduce}'` not yet supported")

        if other.requires_grad:
            ctx.save_for_backward(input, input_value)

        other = other.detach()
        if input_value is not None:
            input_value = input_value.detach()

        if other.is_cuda and torch_geometric.typing.WITH_TORCH_SPARSE:
            # If `torch-sparse` is available, it still provides a faster
            # sparse-dense matmul code path (after all these years...):
            rowptr, col = input.get_rowptr(), input[1]
            return torch.ops.torch_sparse.spmm_sum(  #
                None, rowptr, col, input_value, None, None, other)

        if input_value is None:
            input_value = input._get_value()
        adj = to_sparse_csr(input, input_value)

        return adj @ other

    @staticmethod
    def backward(
        ctx,
        out_grad: Tensor,
    ) -> Tuple[None, Optional[Tensor], None, None]:

        other_grad: Optional[Tensor] = None
        if ctx.needs_input_grad[1]:
            input, input_value = ctx.saved_tensors

            # We need to compute `adj.t() @ out_grad`. For the transpose, we
            # first sort by column and then create a CSR matrix from it.
            # We can call `input.flip(0)` here and create a sparse CSR matrix
            # from it, but creating it via `colptr` directly is more efficient:
            # Note that the sort result is cached across multiple applications.
            input, perm = input.sort_by(SortOrder.COL)
            colptr, row = input.get_colptr(), input[0]
            if input_value is not None:
                input_value = input_value.detach()[perm]

            if out_grad.is_cuda and torch_geometric.typing.WITH_TORCH_SPARSE:
                other_grad = torch.ops.torch_sparse.spmm_sum(  #
                    None, colptr, row, input_value, None, None, out_grad)

            else:
                if input_value is None:
                    input_value = input._get_value()

                adj_t = torch.sparse_csr_tensor(
                    crow_indices=input.get_colptr(),
                    col_indices=input[0],
                    values=input_value,
                    size=input.get_sparse_size()[::-1],
                    device=input.device,
                )

                other_grad = adj_t @ out_grad

        if ctx.needs_input_grad[2]:
            raise NotImplementedError("Gradient computation for 'input_value' "
                                      "not yet supported")

        return None, other_grad, None, None


@implements(torch.matmul)
@implements(Tensor.matmul)
def matmul(
    input: EdgeIndex,
    other: Union[Tensor, EdgeIndex],
    input_value: Optional[Tensor] = None,
    other_value: Optional[Tensor] = None,
    reduce: ReduceType = 'sum',
) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:

    if reduce not in ReduceType.__args__:
        raise NotImplementedError(f"`reduce='{reduce}'` not yet supported")

    if not isinstance(other, EdgeIndex):
        if other_value is not None:
            raise ValueError("'other_value' not supported for sparse-dense "
                             "matrix multiplication")
        return SparseDenseMatmul.apply(input, other, input_value, reduce)

    if input._sort_order == SortOrder.COL:
        input = to_sparse_csc(input, input_value)
    else:
        input = to_sparse_csr(input, input_value)

    if other._sort_order == SortOrder.COL:
        other = to_sparse_csc(other, other_value)
    else:
        other = to_sparse_csr(other, other_value)

    out = torch.matmul(input, other)
    assert out.layout == torch.sparse_csr

    rowptr, col = out.crow_indices(), out.col_indices()
    edge_index = torch._convert_indices_from_csr_to_coo(
        rowptr, col, out_int32=rowptr.dtype != torch.int64)

    edge_index = edge_index.as_subclass(EdgeIndex)
    edge_index._sort_order = SortOrder.ROW
    edge_index._sparse_size = (out.size(0), out.size(1))
    edge_index._rowptr = rowptr

    return edge_index, out.values()


@implements(torch.sparse.mm)
def _matmul(
    mat1: EdgeIndex,
    mat2: Union[Tensor, EdgeIndex],
    reduce: ReduceType = 'sum',
) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:
    return matmul(mat1, mat2, reduce=reduce)
