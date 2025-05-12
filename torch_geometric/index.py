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

import numpy as np
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
    r"""A one-dimensional :obj:`index` tensor with additional (meta)data
    attached.

    :class:`Index` is a :pytorch:`null` :class:`torch.Tensor` that holds
    indices of shape :obj:`[num_indices]`.

    While :class:`Index` sub-classes a general :pytorch:`null`
    :class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

    * :obj:`dim_size`: The size of the underlying sparse vector size, *i.e.*,
      the size of a dimension that can be indexed via :obj:`index`.
      By default, it is inferred as :obj:`dim_size=index.max() + 1`.
    * :obj:`is_sorted`: Whether indices are sorted in ascending order.

    Additionally, :class:`Index` caches data via :obj:`indptr` for fast CSR
    conversion in case its representation is sorted.
    Caches are filled based on demand (*e.g.*, when calling
    :meth:`Index.get_indptr`), or when explicitly requested via
    :meth:`Index.fill_cache_`, and are maintaned and adjusted over its
    lifespan.

    This representation ensures optimal computation in GNN message passing
    schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
    workflows.

    .. code-block:: python

        from torch_geometric import Index

        index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
        >>> Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
        assert index.dim_size == 3
        assert index.is_sorted

        # Flipping order:
        index.flip(0)
        >>> Index([[2, 1, 1, 0], dim_size=3)
        assert not index.is_sorted

        # Filtering:
        mask = torch.tensor([True, True, True, False])
        index[:, mask]
        >>> Index([[0, 1, 1], dim_size=3, is_sorted=True)
        assert index.is_sorted
    """
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
        r"""Validates the :class:`Index` representation.

        In particular, it ensures that

        * it only holds valid indices.
        * the sort order is correctly set.
        """
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
        r"""The size of the underlying sparse vector."""
        return self._dim_size

    @property
    def is_sorted(self) -> bool:
        r"""Returns whether indices are sorted in ascending order."""
        return self._is_sorted

    @property
    def dtype(self) -> torch.dtype:  # type: ignore
        # TODO Remove once PyTorch does not override `dtype` in `DataLoader`.
        return self._data.dtype

    # Cache Interface #########################################################

    def get_dim_size(self) -> int:
        r"""The size of the underlying sparse vector.
        Automatically computed and cached when not explicitly set.
        """
        if self._dim_size is None:
            dim_size = int(self._data.max()) + 1 if self.numel() > 0 else 0
            self._dim_size = dim_size

        assert isinstance(self._dim_size, int)
        return self._dim_size

    def dim_resize_(self, dim_size: Optional[int]) -> 'Index':
        r"""Assigns or re-assigns the size of the underlying sparse vector."""
        if self.is_sorted and self._indptr is not None:
            if dim_size is None:
                self._indptr = None

            elif self._indptr.numel() - 1 >= dim_size:
                self._indptr = self._indptr[:dim_size + 1]

            else:
                fill_value = self._indptr.new_full(
                    (dim_size - self._indptr.numel() + 1, ),
                    fill_value=self._indptr[-1],  # type: ignore
                )
                self._indptr = torch.cat([self._indptr, fill_value], dim=0)

        self._dim_size = dim_size

        return self

    @assert_sorted
    def get_indptr(self) -> Tensor:
        r"""Returns the compressed index representation in case :class:`Index`
        is sorted.
        """
        if self._indptr is None:
            self._indptr = index2ptr(self._data, self.get_dim_size())

        assert isinstance(self._indptr, Tensor)
        return self._indptr

    def fill_cache_(self) -> 'Index':
        r"""Fills the cache with (meta)data information."""
        self.get_dim_size()

        if self.is_sorted:
            self.get_indptr()

        return self

    # Methods #################################################################

    def share_memory_(self) -> 'Index':
        """"""  # noqa: D419
        self._data.share_memory_()
        if self._indptr is not None:
            self._indptr.share_memory_()
        return self

    def is_shared(self) -> bool:
        """"""  # noqa: D419
        return self._data.is_shared()

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`Index` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self._data

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

    def tolist(self) -> List[Any]:
        """"""  # noqa: D419
        return self._data.tolist()

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """"""  # noqa: D419
        return self._data.numpy(force=force)

    # Helpers #################################################################

    def _shallow_copy(self) -> 'Index':
        out = Index(self._data)
        out._dim_size = self._dim_size
        out._is_sorted = self._is_sorted
        out._indptr = self._indptr
        out._cat_metadata = self._cat_metadata
        return out

    def _clear_metadata(self) -> 'Index':
        self._dim_size = None
        self._is_sorted = False
        self._indptr = None
        self._cat_metadata = None
        return self


def apply_(
    tensor: Index,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Union[Index, Tensor]:

    data = fn(tensor._data, *args, **kwargs)

    if data.dtype not in INDEX_DTYPES:
        return data

    if tensor._data.data_ptr() != data.data_ptr():
        out = Index(data)
    else:  # In-place:
        tensor._data = data
        out = tensor

    # Copy metadata:
    out._dim_size = tensor._dim_size
    out._is_sorted = tensor._is_sorted
    out._cat_metadata = tensor._cat_metadata

    # Convert cache:
    if tensor._indptr is not None:
        out._indptr = fn(tensor._indptr, *args, **kwargs)

    return out


@implements(aten.clone.default)
def _clone(
    tensor: Index,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Index:
    out = apply_(tensor, aten.clone.default, memory_format=memory_format)
    assert isinstance(out, Index)
    return out


@implements(aten._to_copy.default)
def _to_copy(
    tensor: Index,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Union[Index, Tensor]:
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
def _alias(tensor: Index) -> Index:
    return tensor._shallow_copy()


@implements(aten._pin_memory.default)
def _pin_memory(tensor: Index) -> Index:
    out = apply_(tensor, aten._pin_memory.default)
    assert isinstance(out, Index)
    return out


@implements(aten.sort.default)
def _sort(
    tensor: Index,
    dim: int = -1,
    descending: bool = False,
) -> Tuple[Index, Tensor]:

    if tensor.is_sorted and not descending:
        return tensor, torch.arange(tensor._data.numel(),
                                    device=tensor._data.device)

    data, perm = aten.sort.default(tensor._data, dim, descending)

    out = Index(data)
    out._dim_size = tensor._dim_size

    if not descending:
        out._is_sorted = True

    return out, perm


@implements(aten.sort.stable)
def _sort_stable(
    tensor: Index,
    *,
    stable: bool = False,
    dim: int = -1,
    descending: bool = False,
) -> Tuple[Index, Tensor]:

    if tensor.is_sorted and not descending:
        return tensor, torch.arange(tensor._data.numel(),
                                    device=tensor._data.device)

    data, perm = aten.sort.stable(tensor._data, stable=stable, dim=dim,
                                  descending=descending)

    out = Index(data)
    out._dim_size = tensor._dim_size

    if not descending:
        out._is_sorted = True

    return out, perm


@implements(aten.cat.default)
def _cat(
    tensors: List[Union[Index, Tensor]],
    dim: int = 0,
) -> Union[Index, Tensor]:

    data_list = pytree.tree_map_only(Index, lambda x: x._data, tensors)
    data = aten.cat.default(data_list, dim=dim)

    if any([not isinstance(tensor, Index) for tensor in tensors]):
        return data

    out = Index(data)

    nnz_list = [t.numel() for t in tensors]
    dim_size_list = [t.dim_size for t in tensors]  # type: ignore
    is_sorted_list = [t.is_sorted for t in tensors]  # type: ignore

    # Post-process `dim_size`:
    total_dim_size: Optional[int] = 0
    for dim_size in dim_size_list:
        if dim_size is None:
            total_dim_size = None
            break
        assert isinstance(total_dim_size, int)
        total_dim_size = max(dim_size, total_dim_size)

    out._dim_size = total_dim_size

    out._cat_metadata = CatMetadata(
        nnz=nnz_list,
        dim_size=dim_size_list,
        is_sorted=is_sorted_list,
    )

    return out


@implements(aten.flip.default)
def _flip(
    input: Index,
    dims: Union[List[int], Tuple[int, ...]],
) -> Index:

    data = aten.flip.default(input._data, dims)

    out = Index(data)
    out._dim_size = input.dim_size

    return out


@implements(aten.index_select.default)
def _index_select(
    input: Union[Index, Tensor],
    dim: int,
    index: Union[Index, Tensor],
) -> Union[Index, Tensor]:

    out = aten.index_select.default(
        input._data if isinstance(input, Index) else input,
        dim,
        index._data if isinstance(index, Index) else index,
    )

    if isinstance(input, Index):
        out = Index(out)
        out._dim_size = input.dim_size

    return out


@implements(aten.slice.Tensor)
def _slice(
    input: Index,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> Index:

    if ((start is None or start <= 0 or start <= -input.size(dim))
            and (end is None or end > input.size(dim)) and step == 1):
        return input._shallow_copy()  # No-op.

    data = aten.slice.Tensor(input._data, dim, start, end, step)

    if step != 1:
        data = data.contiguous()

    out = Index(data)
    out._dim_size = input.dim_size
    # NOTE We could potentially maintain the `indptr` attribute here,
    # but it is not really clear if this is worth it. The most important
    # information `is_sorted` needs to be maintained though:
    if step >= 0:
        out._is_sorted = input.is_sorted

    return out


@implements(aten.index.Tensor)
def _index(
    input: Union[Index, Tensor],
    indices: List[Optional[Union[Tensor, Index]]],
) -> Union[Index, Tensor]:

    if not isinstance(input, Index):
        indices = pytree.tree_map_only(Index, lambda x: x._data, indices)
        return aten.index.Tensor(input, indices)

    data = aten.index.Tensor(input._data, indices)

    if data.dim() != 1:
        return data

    assert len(indices) == 1
    index = indices[0]
    assert index is not None

    out = Index(data)

    if index.dtype in (torch.bool, torch.uint8):  # 1. `index[mask]`.
        out._dim_size = input.dim_size
        out._is_sorted = input.is_sorted

    else:  # 2. `index[index]`.
        out._dim_size = input.dim_size

    return out


@implements(aten.add.Tensor)
def _add(
    input: Union[int, Tensor, Index],
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Union[Index, Tensor]:

    data = aten.add.Tensor(
        input._data if isinstance(input, Index) else input,
        other._data if isinstance(other, Index) else other,
        alpha=alpha,
    )

    if data.dtype not in INDEX_DTYPES:
        return data
    if data.dim() != 1:
        return data

    out = Index(data)

    if isinstance(input, Tensor) and input.numel() <= 1:
        input = int(input)

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        assert isinstance(input, Index)
        if input.dim_size is not None:
            out._dim_size = input.dim_size + alpha * other
        out._is_sorted = input.is_sorted

    elif isinstance(input, int):
        assert isinstance(other, Index)
        if other.dim_size is not None:
            out._dim_size = input + alpha * other.dim_size
        out._is_sorted = other.is_sorted

    elif isinstance(input, Index) and isinstance(other, Index):
        if input.dim_size is not None and other.dim_size is not None:
            out._dim_size = input.dim_size + alpha * other.dim_size

    return out


@implements(aten.add_.Tensor)
def add_(
    input: Index,
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Index:

    dim_size = input.dim_size
    is_sorted = input.is_sorted
    input._clear_metadata()

    aten.add_.Tensor(
        input._data,
        other._data if isinstance(other, Index) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        if dim_size is not None:
            input._dim_size = dim_size + alpha * other
        input._is_sorted = is_sorted

    elif isinstance(other, Index):
        if dim_size is not None and other.dim_size is not None:
            input._dim_size = dim_size + alpha * other.dim_size

    return input


@implements(aten.sub.Tensor)
def _sub(
    input: Union[int, Tensor, Index],
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Union[Index, Tensor]:

    data = aten.sub.Tensor(
        input._data if isinstance(input, Index) else input,
        other._data if isinstance(other, Index) else other,
        alpha=alpha,
    )

    if data.dtype not in INDEX_DTYPES:
        return data
    if data.dim() != 1:
        return data

    out = Index(data)

    if not isinstance(input, Index):
        return out

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        if input.dim_size is not None:
            out._dim_size = input.dim_size - alpha * other
        out._is_sorted = input.is_sorted

    return out


@implements(aten.sub_.Tensor)
def sub_(
    input: Index,
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Index:

    dim_size = input.dim_size
    is_sorted = input.is_sorted
    input._clear_metadata()

    aten.sub_.Tensor(
        input._data,
        other._data if isinstance(other, Index) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        if dim_size is not None:
            input._dim_size = dim_size - alpha * other
        input._is_sorted = is_sorted

    return input
