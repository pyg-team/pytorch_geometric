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


def assert_symmetric(size: Tuple[Optional[int], Optional[int]]):
    if size[0] is not None and size[1] is not None and size[0] != size[1]:
        raise ValueError("'EdgeIndex' is undirected but received a "
                         "non-symmetric size")


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
    * :obj:`is_undirected`: Whether edges are bidirectional.

    Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
    in case its representation is sorted, such as its :obj:`rowptr` or
    :obj:`colptr`, or the permutation vectors from CSR to CSC and vice versa.
    Caches are filled based on demand (*e.g.*, when calling
    :meth:`EdgeIndex.sort_by`), or when explicitly requested via
    :meth:`EdgeIndex.fill_cache_`, and are maintained and adjusted over its
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

    # Whether the `edge_index` is undirected:
    # NOTE `is_undirected` allows us to assume symmetric adjacency matrix size
    # and to share compressed pointer representations, however, it does not
    # allow us get rid of CSR/CSC permutation vectors since ordering within
    # neighborhoods is not necessarily deterministic.
    _is_undirected: bool = False

    # An data cache for CSR and CSC representations:
    _rowptr: Optional[Tensor] = None
    _csr_col: Optional[Tensor] = None

    _colptr: Optional[Tensor] = None
    _csc_row: Optional[Tensor] = None

    _csr2csc: Optional[Tensor] = None
    _csc2csr: Optional[Tensor] = None

    _value: Optional[Tensor] = None  # 1-element value for SpMM.

    def __new__(
        cls,
        data,
        *args,
        sparse_size: Tuple[Optional[int], Optional[int]] = (None, None),
        sort_order: Optional[Union[str, SortOrder]] = None,
        is_undirected: bool = False,
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

        if is_undirected:
            assert_symmetric(sparse_size)
            if sparse_size[0] is not None and sparse_size[1] is None:
                sparse_size = (sparse_size[0], sparse_size[0])
            elif sparse_size[0] is None and sparse_size[1] is not None:
                sparse_size = (sparse_size[1], sparse_size[1])

        out = super().__new__(cls, data)

        # Attach metadata:
        out._sparse_size = sparse_size
        out._sort_order = None if sort_order is None else SortOrder(sort_order)
        out._is_undirected = is_undirected

        return out

    # Validation ##############################################################

    def validate(self) -> 'EdgeIndex':
        r"""Validates the :class:`EdgeIndex` representation, i.e., it ensures
        * that :class:`EdgeIndex` only holds valid entries.
        * that the sort order is correctly set.
        """
        assert_valid_dtype(self)
        assert_two_dimensional(self)
        assert_contiguous(self)
        if self.is_undirected:
            assert_symmetric(self.sparse_size)

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

        if self.is_sorted_by_row and (self[0].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"row indices")

        if self.is_sorted_by_col and (self[1].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"column indices")

        if self.is_undirected:
            flat_index1 = (self[0] * self.get_num_rows() + self[1]).sort()[0]
            flat_index2 = (self[1] * self.get_num_cols() + self[0]).sort()[0]
            if not torch.equal(flat_index1, flat_index2):
                raise ValueError(f"'{self.__class__.__name__}' is not "
                                 f"undirected")

        return self

    # Properties ##############################################################

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

    # Cache Interface #########################################################

    def get_num_rows(self) -> int:
        r"""The number of rows of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        if self.num_rows is None:
            num_rows = int(self[0].max()) + 1 if self.numel() > 0 else 0

            if self.is_undirected:
                self._sparse_size = (num_rows, num_rows)
            else:
                self._sparse_size = (num_rows, self.num_cols)

        return self.num_rows

    def get_num_cols(self) -> int:
        r"""The number of columns of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        if self.num_cols is None:
            num_cols = int(self[1].max()) + 1 if self.numel() > 0 else 0

            if self.is_undirected:
                self._sparse_size = (num_cols, num_cols)
            else:
                self._sparse_size = (self.num_rows, num_cols)

        return self.num_cols

    def get_csr(self) -> Tuple[Tensor, Tensor, Union[Tensor, slice]]:
        if not self.is_sorted:
            raise ValueError(
                f"Cannot access CSR format since '{self.__class__.__name__}' "
                f"is not sorted. Please call `sort_by(...)` first.")

        # For edge indices sorted by row, we just need to compute `rowptr` in
        # case it is not yet populated in the cache.
        if self.is_sorted_by_row:
            # Re-use cache if applicable:
            if self._rowptr is not None:
                rowptr = self._rowptr

            # Re-use CSC cache for undirected edge indices:
            if self.is_undirected and self._colptr is not None:
                rowptr = self._colptr

            else:  # Otherwise, fill cache:
                self._rowptr = torch._convert_indices_from_coo_to_csr(
                    self[0],
                    self.get_num_rows(),
                    out_int32=self.dtype != torch.int64,
                ).to(self.dtype)
                rowptr = self._rowptr

            return rowptr, self[1], slice(None, None, None)

        # For edge indices sorted by column, the logic is a bit more involved.
        # In general, we first compute the CSR permutation, from which we can
        # compute the CSR row and column representations, from which we can
        # then compute the row pointer.
        assert self.is_sorted_by_col

        row: Optional[Tensor] = None
        if self._csc2csr is None:
            row, self._csc2csr = index_sort(self[0], self.get_num_rows())

        if self._csr_col is None:
            self._csr_col = self[1][self._csc2csr]

        rowptr: Optional[Tensor] = self._rowptr
        if rowptr is None:
            if self.is_undirected and self._colptr is not None:
                rowptr = self._colptr

            else:
                if row is None:
                    row = self[0][self._csc2csr]

                self._rowptr = torch._convert_indices_from_coo_to_csr(
                    row,
                    self.get_num_rows(),
                    out_int32=self.dtype != torch.int64,
                ).to(self.dtype)
                rowptr = self._rowptr

        return rowptr, self._csr_col, self._csc2csr

    def get_csc(self) -> Tuple[Tensor, Tensor, Union[Tensor, slice]]:
        if not self.is_sorted:
            raise ValueError(
                f"Cannot access CSC format since '{self.__class__.__name__}' "
                f"is not sorted. Please call `sort_by(...)` first.")

        # For edge indices sorted by column, we just need to compute `colptr`
        # in case it is not yet populated in the cache.
        if self.is_sorted_by_col:
            # Re-use cache if applicable:
            if self._colptr is not None:
                colptr = self._colptr

            # Re-use CSR cache for undirected edge indices:
            if self.is_undirected and self._rowptr is not None:
                colptr = self._rowptr

            else:  # Otherwise, fill cache:
                self._colptr = torch._convert_indices_from_coo_to_csr(
                    self[1],
                    self.get_num_cols(),
                    out_int32=self.dtype != torch.int64,
                ).to(self.dtype)
                colptr = self._colptr

            return colptr, self[0], slice(None, None, None)

        # For edge indices sorted by row, the logic is a bit more involved.
        # In general, we first compute the CSC permutation, from which we can
        # compute the CSC row and column representations, from which we can
        # then compute the column pointer.
        assert self.is_sorted_by_row

        col: Optional[Tensor] = None
        if self._csr2csc is None:
            col, self._csr2csc = index_sort(self[1], self.get_num_cols())

        if self._csc_row is None:
            self._csc_row = self[0][self._csr2csc]

        colptr: Optional[Tensor] = self._colptr
        if colptr is None:
            if self.is_undirected and self._rowptr is not None:
                colptr = self._rowptr

            else:
                if col is None:
                    col = self[1][self._csr2csc]

                self._colptr = torch._convert_indices_from_coo_to_csr(
                    col,
                    self.get_num_cols(),
                    out_int32=self.dtype != torch.int64,
                ).to(self.dtype)
                colptr = self._colptr

        return colptr, self._csc_row, self._csr2csc

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

    def fill_cache_(self, no_transpose: bool = False) -> 'EdgeIndex':
        r"""Fills the cache with (meta)data information.

        Args:
            no_transpose (bool, optional): If set to :obj:`True`, will not fill
                the cache with information about the transposed
                :class:`EdgeIndex`. (default: :obj:`False`)
        """
        self.get_num_rows()
        self.get_num_cols()

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

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`EdgeIndex` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self.as_subclass(Tensor)

    def sort_by(
        self,
        sort_order: Union[str, SortOrder],
    ) -> torch.return_types.sort:
        r"""Sorts the elements by row or column indices.

        Args:
            sort_order (str): The sort order, either :obj:`"row"` or
                :obj:`"col"`.
        """
        sort_order = SortOrder(sort_order)

        if self._sort_order == sort_order:  # Nothing to do.
            return torch.return_types.sort([self, slice(None, None, None)])

        if self.is_sorted_by_row:  # CSR->CSC:
            perm = self._csr2csc

            if (self.is_undirected and perm is not None
                    and self._csc_row is not None):
                edge_index = torch.stack([self._csc_row, self[0]], dim=0)

            elif perm is None:
                col, perm = index_sort(self[1], self.get_num_cols())
                edge_index = torch.stack([self[0][perm], col], dim=0)
                self._csc_row = edge_index[0]
                self._csr2csc = perm
            else:
                edge_index = self.as_tensor()[:, perm]
                self._csc_row = edge_index[0]

        elif self.is_sorted_by_col:  # CSC->CSR:
            perm = self._csc2csr

            if (self.is_undirected and perm is not None
                    and self._csr_col is not None):
                edge_index = torch.stack([self[1], self._csr_col], dim=0)

            elif perm is None:
                row, perm = index_sort(self[0], self.get_num_rows())
                edge_index = torch.stack([row, self[1][perm]], dim=0)
                self._csr_col = edge_index[1]
                self._csc2csr = perm
            else:
                edge_index = self.as_tensor()[:, perm]
                self._csr_col = edge_index[1]

        # Otherwise, perform sorting:
        elif sort_order == SortOrder.ROW:
            row, perm = index_sort(self[0], self.get_num_rows())
            edge_index = torch.stack([row, self[1][perm]], dim=0)

        else:
            col, perm = index_sort(self[1], self.get_num_cols())
            edge_index = torch.stack([self[0][perm], col], dim=0)

        out = self.__class__(edge_index)

        # We can fully inherit metadata and cache:
        out._sparse_size = self.sparse_size
        out._sort_order = sort_order
        out._is_undirected = self.is_undirected

        out._rowptr = self._rowptr
        out._csr_col = self._csr_col

        out._colptr = self._colptr
        out._csc_row = self._csc_row

        out._csr2csc = self._csr2csc
        out._csc2csr = self._csc2csr

        out._value = self._value

        return torch.return_types.sort([out, perm])

    def to_dense(
        self,
        value: Optional[Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:

        # TODO Respect duplicate edges.

        dtype = value.dtype if value is not None else dtype

        size = self.get_sparse_size()
        if value is not None and value.dim() > 1:
            size = size + value.shape[1:]

        out = torch.zeros(size, dtype=dtype, device=self.device)
        out[self[0], self[1]] = value if value is not None else 1

        return out

    def to_sparse_coo(self, value: Optional[Tensor] = None) -> Tensor:
        value = self._get_value() if value is None else value
        out = torch.sparse_coo_tensor(
            indices=self.as_tensor(),
            values=value,
            size=self.get_sparse_size(),
            device=self.device,
            requires_grad=value.requires_grad,
        )

        if self.is_sorted_by_row:
            out = out._coalesced_(True)

        return out

    def to_sparse_csr(self, value: Optional[Tensor] = None) -> Tensor:
        rowptr, col, perm = self.get_csr()
        value = self._get_value() if value is None else value[perm]

        return torch.sparse_csr_tensor(
            crow_indices=rowptr,
            col_indices=col,
            values=value,
            size=self.get_sparse_size(),
            device=self.device,
            requires_grad=value.requires_grad,
        )

    def to_sparse_csc(self, value: Optional[Tensor] = None) -> Tensor:
        if not torch_geometric.typing.WITH_PT112:
            raise NotImplementedError(
                "'to_sparse_csc' not supported for PyTorch < 1.12")

        colptr, row, perm = self.get_csc()
        value = self._get_value() if value is None else value[perm]

        return torch.sparse_csc_tensor(
            ccol_indices=colptr,
            row_indices=row,
            values=value,
            size=self.get_sparse_size(),
            device=self.device,
            requires_grad=value.requires_grad,
        )

    def to_sparse(
        self,
        *,
        layout: Optional[torch.layout] = None,
        value: Optional[Tensor] = None,
    ) -> Tensor:

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
            rowptr=self._rowptr,
            value=value,
            sparse_sizes=self.get_sparse_size(),
            is_sorted=self.is_sorted_by_row,
            trust_data=True,
        )

    def matmul(
        self,
        other: Union[Tensor, 'EdgeIndex'],
        input_value: Optional[Tensor] = None,
        other_value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
    ) -> Union[Tensor, Tuple['EdgeIndex', Tensor]]:
        return matmul(self, other, input_value, other_value, reduce)

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
    out._is_undirected = tensor.is_undirected

    # Convert cache (but do not consider `_value`):
    if tensor._rowptr is not None:
        out._rowptr = fn(tensor._rowptr, *args, **kwargs)
    if tensor._csr_col is not None:
        out._csr_col = fn(tensor._csr_col, *args, **kwargs)

    if tensor._colptr is not None:
        out._colptr = fn(tensor._colptr, *args, **kwargs)
    if tensor._csc_row is not None:
        out._csc_row = fn(tensor._csc_row, *args, **kwargs)

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

    if len(tensors) == 1:
        return tensors[0]

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

    # Post-process `is_undirected`:
    is_undirected = True
    for tensor in tensors:
        is_undirected = tensor.is_undirected

    out._is_undirected = is_undirected

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

    out._value = input._value
    out._is_undirected = input.is_undirected

    # Flip metadata and cache:
    if 0 in dims or -2 in dims:
        out._sparse_size = input.sparse_size[::-1]

    if len(dims) == 1 and (dims[0] == 0 or dims[0] == -2):
        if input.is_sorted_by_row:
            out._sort_order = SortOrder.COL
        elif input.is_sorted_by_col:
            out._sort_order = SortOrder.ROW

        out._rowptr = input._colptr
        out._csr_col = input._csc_row

        out._colptr = input._rowptr
        out._csc_row = input._csr_col

        out._csr2csc = input._csc2csr
        out._csc2csr = input._csr2csc

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


# Sparse-Dense Matrix Multiplication ##########################################


def _torch_sparse_spmm(
    input: EdgeIndex,
    other: Tensor,
    value: Optional[Tensor] = None,
    reduce: ReduceType = 'sum',
) -> Tensor:
    # `torch-sparse` still provides a faster sparse-dense matrix multiplication
    # code path on GPUs (after all these years...):
    assert input.is_sorted_by_row
    assert torch_geometric.typing.WITH_TORCH_SPARSE

    rowptr = input.get_rowptr()
    col = input[1]

    # Optional arguments for backpropagation:
    row: Optional[Tensor] = None
    rowcount: Optional[Tensor] = None
    colptr: Optional[Tensor] = None
    csr2csc: Optional[Tensor] = None

    if reduce == 'sum':
        if value is not None and value.requires_grad:
            row = input[0]
        if other.requires_grad:
            row = input[0]
            csr2csc = input._get_csr2csc()
            colptr = input.get_colptr()
        return torch.ops.torch_sparse.spmm_sum(  #
            row, rowptr, col, value, colptr, csr2csc, other)

    if reduce == 'mean':
        if value is not None and value.requires_grad:
            row = input[0]
        if other.requires_grad:
            row = input[0]
            rowcount = rowptr.diff()
            csr2csc = input._get_csr2csc()
            colptr = input.get_colptr()
        return torch.ops.torch_sparse.spmm_mean(  #
            row, rowptr, col, value, rowcount, colptr, csr2csc, other)

    if reduce == 'amin':
        return torch.ops.torch_sparse.spmm_min(rowptr, col, value, other)

    if reduce == 'amax':
        return torch.ops.torch_sparse.spmm_max(rowptr, col, value, other)

    raise NotImplementedError


class SparseDenseMatmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: EdgeIndex,
        other: Tensor,
        input_value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
    ) -> Tensor:

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
            rowptr, col, perm = input.get_csr()
            if input_value is not None:
                input_value = input_value[perm]
            return torch.ops.torch_sparse.spmm_sum(  #
                None, rowptr, col, input_value, None, None, other)

        if input_value is None:
            input_value = input._get_value()
        adj = input.to_sparse_csr(input_value)

        return adj @ other

    @staticmethod
    def backward(
        ctx,
        out_grad: Tensor,
    ) -> Tuple[None, Optional[Tensor], None, None]:

        # TODO Leverage `is_undirected` in case `input_value` is None.

        other_grad: Optional[Tensor] = None
        if ctx.needs_input_grad[1]:
            input, input_value = ctx.saved_tensors

            # We need to compute `adj.t() @ out_grad`. For the transpose, we
            # first sort by column and then create a CSR matrix from it.
            # We can call `input.flip(0)` here and create a sparse CSR matrix
            # from it, but creating it via `colptr` directly is more efficient:
            # Note that the sort result is cached across multiple applications.
            colptr, row, perm = input.get_csc()

            if input_value is not None:
                input_value = input_value.detach()[perm]

            if out_grad.is_cuda and torch_geometric.typing.WITH_TORCH_SPARSE:
                other_grad = torch.ops.torch_sparse.spmm_sum(  #
                    None, colptr, row, input_value, None, None, out_grad)

            else:
                if input_value is None:
                    input_value = input._get_value()

                adj_t = torch.sparse_csr_tensor(
                    crow_indices=colptr,
                    col_indices=row,
                    values=input_value,
                    size=input.get_sparse_size()[::-1],
                    device=input.device,
                )

                other_grad = adj_t @ out_grad

        if ctx.needs_input_grad[2]:
            raise NotImplementedError("Gradient computation for 'input_value' "
                                      "not yet supported")

        return None, other_grad, None, None


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

    if input.is_sorted_by_col:
        input = input.to_sparse_csc(input_value)
    else:
        input = input.to_sparse_csr(input_value)

    if other.is_sorted_by_col:
        other = other.to_sparse_csc(other_value)
    else:
        other = other.to_sparse_csr(other_value)

    out = torch.matmul(input, other)
    assert out.layout == torch.sparse_csr

    rowptr, col = out.crow_indices(), out.col_indices()
    edge_index = torch._convert_indices_from_csr_to_coo(
        rowptr, col, out_int32=rowptr.dtype != torch.int64)
    edge_index = edge_index.to(rowptr.device)

    edge_index = edge_index.as_subclass(EdgeIndex)
    edge_index._sort_order = SortOrder.ROW
    edge_index._sparse_size = (out.size(0), out.size(1))
    edge_index._rowptr = rowptr

    return edge_index, out.values()


@implements(torch.matmul)
@implements(Tensor.matmul)
def _matmul1(
    input: EdgeIndex,
    other: Union[Tensor, EdgeIndex],
) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:
    return matmul(input, other)


@implements(torch.sparse.mm)
def _matmul2(
    mat1: EdgeIndex,
    mat2: Union[Tensor, EdgeIndex],
    reduce: ReduceType = 'sum',
) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:
    return matmul(mat1, mat2, reduce=reduce)
