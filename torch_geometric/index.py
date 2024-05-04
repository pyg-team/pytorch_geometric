import functools
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union

import torch
from torch import Tensor

from torch_geometric.typing import INDEX_DTYPES

aten = torch.ops.aten

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def ptr2index(ptr: Tensor, output_size: Optional[int] = None) -> Tensor:
    index = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return index.repeat_interleave(ptr.diff(), output_size=output_size)


def index2ptr(index: Tensor, size: Optional[int] = None) -> Tensor:
    if size is None:
        size = int(index.max()) + 1 if index.numel() > 0 else 0

    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype != torch.int64)


class CatMetadata(NamedTuple):
    nnz: List[int]
    dim_size: List[Optional[int]]
    is_sorted: List[bool]


def implements(torch_function: Callable) -> Callable:
    r"""Registers a :pytorch:`PyTorch` function override."""
    @functools.wraps(torch_function)
    def decorator(my_function: Callable) -> Callable:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


def assert_valid_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in INDEX_DTYPES:
        raise ValueError(f"'Index' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{INDEX_DTYPES})")


def assert_one_dimensional(tensor: Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"'Index' needs to be one-dimensional "
                         f"(got {tensor.dim()} dimensions)")


def assert_contiguous(tensor: Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError("'Index' needs to be contiguous. Please call "
                         "`index.contiguous()` before proceeding.")


def assert_sorted(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: 'Index', *args: Any, **kwargs: Any) -> Any:
        if not self.is_sorted:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"Cannot call '{func.__name__}' since '{cls_name}' is not "
                f"sorted. Please call `{cls_name}.sort()` first.")
        return func(self, *args, **kwargs)

    return wrapper


class Index(Tensor):
    r"""TODO."""
    # See "https://pytorch.org/docs/stable/notes/extending.html"
    # for a basic tutorial on how to subclass `torch.Tensor`.

    # The underlying tensor representation:
    _data: Tensor

    # The size of the underlying sparse vector, e.g. `_data.max() + 1` :
    _dim_size: Optional[int] = None

    # Whether the `index` representation is sorted:
    _is_sorted: bool = False

    # A cache for its compressed representation:
    _indptr: Optional[Tensor] = None

    # Whenever we perform a concatenation of indices, we cache the original
    # metadata to be able to reconstruct individual indices:
    _cat_metadata: Optional[CatMetadata] = None

    @staticmethod
    def __new__(
        cls: Type,
        data: Any,
        *args: Any,
        dim_size: Optional[int] = None,
        is_sorted: bool = False,
        **kwargs: Any,
    ) -> 'Index':
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

        if isinstance(data, cls):  # If passed `Index`, inherit metadata:
            indptr = data._indptr
            dim_size = dim_size or data.dim_size()
            is_sorted = is_sorted or data.is_sorted

        assert_valid_dtype(data)
        assert_one_dimensional(data)
        assert_contiguous(data)

        out = Tensor._make_wrapper_subclass(  # type: ignore
            cls,
            size=data.size(),
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            layout=data.layout,
            requires_grad=False,
        )
        assert isinstance(out, Index)

        # Attach metadata:
        out._data = data
        out._dim_size = dim_size
        out._is_sorted = is_sorted
        out._indptr = indptr

        if isinstance(data, cls):
            out._data = data._data

            # Reset metadata if cache is invalidated:
            if dim_size is not None and dim_size != data.dim_size():
                out._indptr = None

        return out

    # Validation ##############################################################

    def validate(self) -> 'Index':
        raise NotImplementedError

    # Properties ##############################################################

    @property
    def dim_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def is_sorted(self) -> bool:
        raise NotImplementedError

    # Cache Interface #########################################################

    def get_dim_size(self) -> int:
        raise NotImplementedError

    def dim_resize_(self, dim_size: Optional[int]) -> 'Index':
        raise NotImplementedError

    @assert_sorted
    def get_indptr(self) -> Tensor:
        raise NotImplementedError

    def fill_cache_(self) -> 'Index':
        raise NotImplementedError

    # Methods #################################################################

    def share_memory_(self) -> 'Index':
        self._data.share_memory_()
        if self._indptr is not None:
            self._indptr.share_memory_()
        return self

    def is_shared(self) -> bool:
        return self._data.is_shared()

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`Index` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self._data

    def dim_narrow(self, start: Union[int, Tensor], length: int) -> 'Index':
        raise NotImplementedError


# def sort(self) -> None:
#     # TODO MOVE BEHIND TORCH DISPATCH
#     raise NotImplementedError
