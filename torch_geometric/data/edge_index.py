from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor


class SortOrder(Enum):
    ROW = 'row'
    COL = 'col'


class EdgeIndex(Tensor):
    _sparse_size: Tuple[Optional[int], Optional[int]] = (None, None)
    _sort_order: Optional[SortOrder] = None

    _rowptr: Optional[Tensor] = None
    _colptr: Optional[Tensor] = None
    _csr2csc: Optional[Tensor] = None
    _csc2csr: Optional[Tensor] = None

    def __new__(
        cls,
        data,
        *args,
        sparse_size: Tuple[Optional[int], Optional[int]] = (None, None),
        sort_order: Optional[SortOrder] = None,
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

    @property
    def sparse_size(self) -> Tuple[Optional[int], Optional[int]]:
        return self._sparse_size

    @property
    def sort_order(self) -> Optional[SortOrder]:
        return self._sort_order

    def validate(self) -> bool:
        raise NotImplementedError  # TODO

    def to_tensor(self) -> Tensor:
        return torch.as_tensor(self)  # TODO

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # TODO Disallow in-place operations?
        return super().__torch_function__(func, types, args, kwargs)
