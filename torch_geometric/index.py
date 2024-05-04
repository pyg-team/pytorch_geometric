import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.utils._pytree as pytree
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
            dim_size = dim_size or data.dim_size
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
            if dim_size is not None and dim_size != data.dim_size:
                out._indptr = None

        return out

    # Validation ##############################################################

    def validate(self) -> 'Index':
        r"""TODO."""
        assert_valid_dtype(self._data)
        assert_one_dimensional(self._data)
        assert_contiguous(self._data)

        if self.numel() > 0 and self._data.min() < 0:
            raise ValueError(f"'{self.__class__.__name__}' contains negative "
                             f"indices (got {int(self.min())})")

        if (self.numel() > 0 and self.dim_size is not None
                and self._data.max() >= self.dim_size):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its registered size "
                             f"(got {int(self._data.max())}, but expected "
                             f"values smaller than {self.dim_size})")

        if self.is_sorted and (self._data.diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted")

        return self

    # Properties ##############################################################

    @property
    def dim_size(self) -> Optional[int]:
        r"""TODO."""
        return self._dim_size

    @property
    def is_sorted(self) -> bool:
        r"""TODO."""
        return self._is_sorted

    # Cache Interface #########################################################

    def get_dim_size(self) -> int:
        r"""TODO."""
        if self._dim_size is None:
            dim_size = int(self._data.max()) + 1 if self.numel() > 0 else 0
            self._dim_size = dim_size

        assert isinstance(self._dim_size, int)
        return self._dim_size

    def dim_resize_(self, dim_size: Optional[int]) -> 'Index':
        r"""TODO."""
        raise NotImplementedError  # TODO

    @assert_sorted
    def get_indptr(self) -> Tensor:
        r"""TODO."""
        if self._indptr is None:
            self._indptr = index2ptr(self._data, self.get_dim_size())

        assert isinstance(self._indptr, Tensor)
        return self._indptr

    def fill_cache_(self) -> 'Index':
        r"""TODO."""
        self.get_dim_size()

        if self.is_sorted:
            self.get_indptr()

        return self

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
        r"""TODO."""
        raise NotImplementedError  # TODO

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[Any, ...]]:
        attrs = ['_data']
        if self._indptr is not None:
            attrs.append('_indptr')

        ctx = (
            self._dim_size,
            self._is_sorted,
            self._cat_metadata,
        )

        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, Any],
        ctx: Tuple[Any, ...],
        outer_size: Tuple[int, ...],
        outer_stride: Tuple[int, ...],
    ) -> 'Index':
        index = Index(
            inner_tensors['_data'],
            dim_size=ctx[0],
            is_sorted=ctx[1],
        )

        index._indptr = inner_tensors.get('_indptr', None)
        index._cat_metadata = ctx[2]

        return index

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
        # `Index` should be treated as a regular PyTorch tensor for all
        # standard PyTorch functionalities. However,
        # * some of its metadata can be transferred to new functions, e.g.,
        #   `torch.narrow()` can inherit the `is_sorted` property.
        # * not all operations lead to valid `Index` tensors again, e.g.,
        #   `torch.sum()` does not yield a `Index` as its output, or
        #   `torch.stack() violates the [*] shape assumption.

        # To account for this, we hold a number of `HANDLED_FUNCTIONS` that
        # implement specific functions for valid `Index` routines.
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        # For all other PyTorch functions, we treat them as vanilla tensors.
        args = pytree.tree_map_only(Index, lambda x: x._data, args)
        if kwargs is not None:
            kwargs = pytree.tree_map_only(Index, lambda x: x._data, kwargs)
        return func(*args, **(kwargs or {}))

    def __repr__(self) -> str:  # type: ignore
        prefix = f'{self.__class__.__name__}('
        indent = len(prefix)
        tensor_str = torch._tensor_str._tensor_str(self._data, indent)

        suffixes = []
        if self.dim_size is not None:
            suffixes.append(f'dim_size={self.dim_size}')
        if (self.device.type != torch._C._get_default_device()
                or (self.device.type == 'cuda'
                    and torch.cuda.current_device() != self.device.index)
                or (self.device.type == 'mps')):
            suffixes.append(f"device='{self.device}'")
        if self.dtype != torch.int64:
            suffixes.append(f'dtype={self.dtype}')
        if self.is_sorted:
            suffixes.append('is_sorted=True')

        return torch._tensor_str._add_suffixes(prefix + tensor_str, suffixes,
                                               indent, force_newline=False)


# def sort(self) -> None:
#     # TODO MOVE BEHIND TORCH DISPATCH
#     raise NotImplementedError
