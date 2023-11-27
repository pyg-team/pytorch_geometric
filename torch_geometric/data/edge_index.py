import functools
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def implements(torch_function: Callable):
    r"""Registers a :pytorch:`PyTorch` function override."""
    @functools.wraps(torch_function)
    def decorator(my_function: Callable):
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


def assert_integral_type(tensor: Tensor):
    if tensor.is_floating_point():
        raise ValueError(f"'EdgeIndex' needs to be of integral type "
                         f"(got '{tensor.dtype}')")


def assert_two_dimensional(tensor: Tensor):
    if tensor.dim() != 2:
        raise ValueError(f"'EdgeIndex' needs to be two-dimensional "
                         f"(got {tensor.dim()} dimensions)")
    if tensor.size(0) != 2:
        raise ValueError(f"'EdgeIndex' needs to have a shape of "
                         f"[2, *] (got {list(tensor.size())})")


class SortOrder(Enum):
    ROW = 'row'
    COL = 'col'


class EdgeIndex(Tensor):
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
        assert_integral_type(data)
        assert_two_dimensional(data)

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
        assert_integral_type(self)
        assert_two_dimensional(self)

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

    def fill_cache(self) -> 'EdgeIndex':
        if self._sort_order == SortOrder.ROW and self._rowptr is None:
            if self.num_rows is None:
                self._sparse_size = (
                    int(self[0].max()) + 1 if self.numel() > 0 else 0,
                    self.num_cols,
                )

            self._rowptr = torch._convert_indices_from_coo_to_csr(
                self[0], self.num_rows, out_int32=self.dtype != torch.int64)
            self._rowptr = self._rowptr.to(self.dtype)

        if self._sort_order == SortOrder.COL and self._colptr is None:
            if self.num_cols is None:
                self._sparse_size = (
                    self.num_rows,
                    int(self[1].max()) + 1 if self.numel() > 0 else 0,
                )

            self._colptr = torch._convert_indices_from_coo_to_csr(
                self[1], self.num_cols, out_int32=self.dtype != torch.int64)
            self._colptr = self._colptr.to(self.dtype)

    @property
    def sparse_size(self) -> Tuple[Optional[int], Optional[int]]:
        r"""The size of the underlying sparse matrix."""
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

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`EdgeIndex` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self.as_subclass(Tensor)

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
    out._sparse_size = out._sparse_size
    out._sort_order = out._sort_order

    # Convert metadata:
    if tensor._rowptr is not None:
        out._rowptr = fn(tensor._rowptr, *args, **kwargs)
    if tensor._colptr is not None:
        out._colptr = fn(tensor._colptr, *args, **kwargs)
    if tensor._csr2csc is not None:
        out._csr2csc = fn(tensor._csr2csc, *args, **kwargs)
    if tensor._csc2csr is not None:
        out._csc2csr = fn(tensor._csc2csr, *args, **kwargs)

    return out


@implements(Tensor.to)
def to(tensor: EdgeIndex, *args, **kwargs) -> EdgeIndex:
    out = apply_(tensor, Tensor.to, *args, **kwargs)
    assert_integral_type(out)
    return out


@implements(Tensor.cpu)
def cpu(tensor: EdgeIndex, *args, **kwargs) -> EdgeIndex:
    return apply_(tensor, Tensor.cpu, *args, **kwargs)


@implements(Tensor.cuda)
def cuda(tensor: EdgeIndex, *args, **kwargs) -> EdgeIndex:
    return apply_(tensor, Tensor.cuda, *args, **kwargs)


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
