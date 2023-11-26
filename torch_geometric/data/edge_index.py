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

        if data.is_floating_point():
            raise ValueError(f"'{cls.__name__}' needs to be of integral type "
                             f"(got '{data.dtype}')")
        if data.dim() != 2:
            raise ValueError(f"'{cls.__name__}' needs to be two-dimensional "
                             f"(got {data.dim()} dimensions)")
        if data.size(0) != 2:
            raise ValueError(f"'{cls.__name__}' needs to have a shape of "
                             f"[2, *] (got {list(data.size())})")

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
        return torch.Tensor.__torch_function__(func, types, args, kwargs)


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
