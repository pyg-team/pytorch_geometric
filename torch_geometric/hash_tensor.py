import functools
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.utils._pytree as pytree
import xxhash
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import CPUHashMap, CUDAHashMap

aten = torch.ops.aten

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def implements(torch_function: Callable) -> Callable:
    r"""Registers a :pytorch:`PyTorch` function override."""
    @functools.wraps(torch_function)
    def decorator(my_function: Callable) -> Callable:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


def as_key_tensor(
    key: Any,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    try:
        key = torch.as_tensor(key, device=device)
    except Exception:
        device = device or torch.get_default_device()
        key = torch.tensor(
            [xxhash.xxh64(x).intdigest() & 0x7FFFFFFFFFFFFFFF for x in key],
            dtype=torch.int64, device=device)

    if key.element_size() == 1:
        key = key.view(torch.uint8)
    elif key.element_size() == 2:
        key = key.view(torch.int16)
    elif key.element_size() == 4:
        key = key.view(torch.int32)
    elif key.element_size() == 8:
        key = key.view(torch.int64)
    else:
        raise ValueError(f"Received invalid dtype '{key.dtype}' with "
                         f"{key.element_size()} bytes")

    return key


def get_hash_map(key: Tensor) -> Union[CPUHashMap, CUDAHashMap]:
    if torch_geometric.typing.WITH_CUDA_HASH_MAP and key.is_cuda:
        return CUDAHashMap(key, 0.5)

    if key.is_cuda:
        warnings.warn("Fallback to CPU-based mapping algorithm which may "
                      "cause slowdowns and device synchronization. Please "
                      "install 'pyg-lib' for an accelerated 'HashTensor' "
                      "implementation.")

    if torch_geometric.typing.WITH_CPU_HASH_MAP:
        return CPUHashMap(key.cpu(), -1)

    import pandas as pd

    return pd.CategoricalDtype(
        categories=key.cpu().numpy(),
        ordered=True,
    )


class HashTensor(Tensor):
    r"""A :pytorch:`null` :class:`torch.Tensor` that can be referenced by
    arbitrary keys rather than indices in the first dimension.

    :class:`HashTensor` sub-classes a general :pytorch:`null`
    :class:`torch.Tensor`, and extends it by CPU- and GPU-accelerated mapping
    routines. This allow for fast and efficient access to non-contiguous
    indices/keys while the underlying data is stored in a compact format.

    This representation is ideal for scenarios where one needs a fast mapping
    routine without relying on CPU-based external packages, and can be used,
    *e.g.*, to perform mapping of global indices to local indices during
    subgraph creation, or in data-processing pipelines to map non-contiguous
    input data into a contiguous space, such as

    * mapping of hashed node IDs to range :obj:`[0, num_nodes - 1]`
    * mapping of raw input data, *e.g.*, categorical data to range
      :obj:`[0, num_categories - 1]`

    Specifically, :class:`HashTensor` supports *any* keys of *any* type,
    *e.g.*, strings, timestamps, etc.

    .. code-block:: python

        from torch_geometric import HashTensor

        key = torch.tensor([1000, 100, 10000])
        value = torch.randn(3, 4)

        tensor = HashTensor(key, value)
        assert tensor.size() == (3, 4)

        # Filtering:
        query = torch.tensor([10000, 1000])
        out = tensor[query]
        assert out.equal(value[[2, 0]])

        # Accessing non-existing keys:
        out = tensor[[10000, 0]]
        out.isnan()
        >>> tensor([[False, False, False, False],
        ...         [True, True, True, True])

        # If `value` is not given, indexing returns the position of `query` in
        # `key`, and `-1` otherwise:
        key = ['Animation', 'Comedy', 'Fantasy']
        tensor = HashTensor(key)

        out = tensor[['Comedy', 'Romance']]
        >>> tensor([1, -1])

    Args:
        key: The keys in the first dimension.
        value: The values to hold.
        dtype: The desired data type of the values of the returned tensor.
        device: The device of the returned tensor.
    """
    _map: Union[Tensor, CPUHashMap, CUDAHashMap]
    _value: Optional[Tensor]
    _min_key: Tensor
    _max_key: Tensor

    @staticmethod
    def __new__(
        cls: Type,
        key: Any,
        value: Optional[Any] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> 'HashTensor':

        if value is not None:
            value = torch.as_tensor(value, dtype=dtype, device=device)
            device = value.device

        key = as_key_tensor(key, device=device)

        if key.dim() != 1:
            raise ValueError(f"'key' data in '{cls.__name__}' needs to be "
                             f"one-dimensional (got {key.dim()} dimensions)")

        if not key.is_contiguous():
            raise ValueError(f"'key' data in '{cls.__name__}' needs to be "
                             f"contiguous")

        if value is not None:
            if key.device != value.device:
                raise ValueError(f"'key' and 'value' data in '{cls.__name__}' "
                                 f"are expected to be on the same device (got "
                                 f"'{key.device}' and '{value.device}')")

            if key.numel() != value.size(0):
                raise ValueError(f"'key' and 'value' data in '{cls.__name__}' "
                                 f"are expected to have the same size in the "
                                 f"first dimension (got {key.size(0)} and "
                                 f"{value.size(0)})")

        min_key = key.min() if key.numel() > 0 else key.new_zeros(())
        max_key = key.max() if key.numel() > 0 else key.new_zeros(())

        _range = max_key - min_key
        # TODO Expose fixed threshold as argument.
        if (key.dtype in {torch.uint8, torch.int16} or _range <= 1_000_000
                or _range <= 2 * key.numel()):
            _map = torch.full(
                size=(_range + 3, ),
                fill_value=-1,
                dtype=torch.int64,
                device=key.device,
            )
            _map[key.long() - (min_key.long() - 1)] = torch.arange(
                key.numel(),
                dtype=_map.dtype,
                device=_map.device,
            )
        else:
            _map = get_hash_map(key)

        return cls._from_data(
            _map,
            value,
            min_key,
            max_key,
            num_keys=key.numel(),
            dtype=dtype,
        )

    # Private Methods #########################################################

    @classmethod
    def _from_data(
        cls,
        _map: Union[Tensor, CPUHashMap, CUDAHashMap],
        value: Optional[Tensor],
        min_key: Tensor,
        max_key: Tensor,
        *,
        num_keys: int,
        dtype: Optional[torch.dtype],
    ) -> 'HashTensor':

        if value is not None:
            dtype = value.dtype
            size = value.size()
            stride = value.stride()
            layout = value.layout
            requires_grad = value.requires_grad
        else:
            dtype = dtype or torch.int64
            size = torch.Size([num_keys])
            stride = (1, )
            layout = torch.strided
            requires_grad = False

        out = Tensor._make_wrapper_subclass(  # type: ignore
            cls,
            size=size,
            strides=stride,
            dtype=dtype,
            device=min_key.device,
            layout=layout,
            requires_grad=requires_grad,
        )
        assert isinstance(out, HashTensor)

        out._map = _map
        out._value = value
        out._min_key = min_key
        out._max_key = max_key

        return out

    @property
    def _key(self) -> Tensor:
        if isinstance(self._map, Tensor):
            mask = self._map >= 0
            key = mask.nonzero().view(-1) - 1
            key = key[self._map[mask]]
        elif (torch_geometric.typing.WITH_CUDA_HASH_MAP
              or torch_geometric.typing.WITH_CPU_HASH_MAP):
            key = self._map.keys().to(self.device)
        else:
            key = torch.from_numpy(self._map.categories.to_numpy())

        return key.to(self.device)

    def _shallow_copy(self) -> 'HashTensor':
        return self._from_data(
            self._map,
            self._value,
            self._min_key,
            self._max_key,
            num_keys=self.size(0),
            dtype=self.dtype,
        )

    def _get(self, query: Tensor) -> Tensor:
        if isinstance(self._map, Tensor):
            index = query.long() - (self._min_key.long() - 1)
            index = self._map[index.clamp_(min=0, max=self._map.numel() - 1)]
        elif torch_geometric.typing.WITH_CUDA_HASH_MAP and query.is_cuda:
            index = self._map.get(query)
        elif torch_geometric.typing.WITH_CPU_HASH_MAP:
            index = self._map.get(query.cpu())
        else:
            import pandas as pd

            ser = pd.Series(query.cpu().numpy(), dtype=self._map)
            index = torch.from_numpy(ser.cat.codes.to_numpy().copy()).long()

        index = index.to(self.device)

        if self._value is None:
            return index.to(self.dtype)

        out = self._value[index]

        mask = index != -1
        mask = mask.view([-1] + [1] * (out.dim() - 1))
        fill_value = float('NaN') if out.is_floating_point() else -1
        if torch_geometric.typing.WITH_PT20:
            other: Union[int, float, Tensor] = fill_value
        else:
            other = torch.full_like(out, fill_value)

        return out.where(mask, other)

    # Methods #################################################################

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`HashTensor` representation back to a
        :class:`torch.Tensor` representation.
        """
        if self._value is not None:
            return self._value
        return torch.arange(self.size(0), dtype=self.dtype, device=self.device)

    # PyTorch/Python builtins #################################################

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
        # Hold a number of `HANDLED_FUNCTIONS` that implement specific
        # functions for valid `HashTensor` routines.
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        # For all other PyTorch functions, we treat them as vanilla tensors.
        args = pytree.tree_map_only(HashTensor, lambda x: x.as_tensor(), args)
        if kwargs is not None:
            kwargs = pytree.tree_map_only(HashTensor, lambda x: x.as_tensor(),
                                          kwargs)
        return func(*args, **(kwargs or {}))

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[Any, ...]]:
        attrs = ['_map', '_min_key', '_max_key']
        if self._value is not None:
            attrs.append('_value')

        ctx = (self.size(0), self.dtype)

        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, Any],
        ctx: Tuple[Any, ...],
        outer_size: Tuple[int, ...],
        outer_stride: Tuple[int, ...],
    ) -> 'HashTensor':
        return HashTensor._from_data(
            inner_tensors['_map'],
            inner_tensors.get('_value', None),
            inner_tensors['_min_key'],
            inner_tensors['_min_key'],
            num_keys=ctx[0],
            dtype=ctx[1],
        )

    def __repr__(self) -> str:  # type: ignore
        indent = len(f'{self.__class__.__name__}(')
        tensor_str = torch._tensor_str._tensor_str(self.as_tensor(), indent)
        return torch._tensor_str._str_intern(self, tensor_contents=tensor_str)

    def tolist(self) -> List[Any]:
        """"""  # noqa: D419
        return self.as_tensor().tolist()

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """"""  # noqa: D419
        return self.as_tensor().numpy(force=force)

    def index_select(  # type: ignore
        self,
        dim: int,
        index: Any,
    ) -> Union['HashTensor', Tensor]:
        """"""  # noqa: D419
        return torch.index_select(self, dim, index)

    def select(  # type: ignore
        self,
        dim: int,
        index: Any,
    ) -> Union['HashTensor', Tensor]:
        """"""  # noqa: D419
        return torch.select(self, dim, index)

    def share_memory_(self) -> 'HashTensor':
        """"""  # noqa: D419
        if isinstance(self._map, Tensor):
            self._map.share_memory_()
        if self._value is not None:
            self._value.share_memory_()
        self._min_key.share_memory_()
        self._max_key.share_memory_()
        return self

    def is_shared(self) -> bool:
        """"""  # noqa: D419
        return self._min_key.is_shared()

    def detach_(self) -> 'HashTensor':  # type: ignore
        """"""  # noqa: D419
        if self._value is not None:
            self._value.detach_()
        return super().detach_()  # type: ignore

    def __getitem__(self, indices: Any) -> Union['HashTensor', Tensor]:
        if not isinstance(indices, tuple):
            indices = (indices, )
        assert len(indices) > 0

        # We convert any index tensor in the first dimension into a tensor.
        # This means that downstream handling (i.e. in `aten.index.Tensor`)
        # needs to take this pre-conversion into account. However, detecting
        # whether the first dimension is indexed can be tricky at times:
        # * We need to take into account `Ellipsis`
        # * We need to take any unsqueezing into account
        if indices[0] is Ellipsis and len(indices) > 1:
            nonempty_indices = [i for i in indices[1:] if i is not None]
            if len(nonempty_indices) == self.dim():
                indices = indices[1:]

        if isinstance(indices[0], (int, bool)):
            index: Union[int, Tensor] = int(as_key_tensor([indices[0]]))
            indices = (index, ) + indices[1:]
        elif isinstance(indices[0], (Tensor, list, np.ndarray)):
            index = as_key_tensor(indices[0], device=self.device)
            indices = (index, ) + indices[1:]

        indices = indices[0] if len(indices) == 1 else indices

        return super().__getitem__(indices)


@implements(aten.alias.default)
def _alias(tensor: HashTensor) -> HashTensor:
    return tensor._shallow_copy()


@implements(aten.clone.default)
def _clone(
    tensor: HashTensor,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> HashTensor:

    value = tensor._value
    if value is not None:
        value = aten.clone.default(value, memory_format=memory_format)

    return tensor._from_data(
        tensor._map,  # NOTE No reason to do clone since it is read-only.
        value,
        tensor._min_key,  # NOTE No reason to do clone since it is read-only.
        tensor._max_key,  # NOTE No reason to do clone since it is read-only.
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


@implements(aten.detach.default)
def _detach(tensor: HashTensor) -> HashTensor:
    value = tensor._value
    if value is not None:
        value = aten.detach.default(value)

    return tensor._from_data(
        tensor._map,
        value,
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


@implements(aten._to_copy.default)
def _to_copy(
    tensor: HashTensor,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> HashTensor:

    value = tensor._value
    if value is not None:
        value = aten._to_copy.default(
            value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    min_key = aten._to_copy.default(tensor._min_key, device=device)
    max_key = aten._to_copy.default(tensor._max_key, device=device)

    _map = tensor._map
    if isinstance(_map, Tensor):
        _map = aten._to_copy.default(_map, device=device)
    # Only convert `_map` in case `CUDAHashMap` exists - otherwise we use
    # CPU-based mapping anyway and there is no need for a copy.
    elif (torch_geometric.typing.WITH_CUDA_HASH_MAP and tensor.is_cuda
          and tensor.device != min_key.device):
        key = _map.keys()
        key = aten._to_copy.default(key, device=device)
        _map = get_hash_map(key)

    return tensor._from_data(
        _map,
        value,
        min_key,
        max_key,
        num_keys=tensor.size(0),
        dtype=dtype or tensor.dtype,
    )


@implements(aten._pin_memory.default)
def _pin_memory(tensor: HashTensor) -> HashTensor:
    _map = tensor._map
    if isinstance(_map, Tensor):
        _map = aten._pin_memory.default(_map)

    value = tensor._value
    if value is not None:
        value = aten._pin_memory.default(value)

    return tensor._from_data(
        _map,
        value,
        aten._pin_memory.default(tensor._min_key),
        aten._pin_memory.default(tensor._max_key),
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


@implements(aten.unsqueeze.default)
def _unsqueeze(tensor: HashTensor, dim: int) -> HashTensor:
    if dim == 0 or dim == -(tensor.dim() + 1):
        raise IndexError(f"Cannot unsqueeze '{tensor.__class__.__name__}' in "
                         f"the first dimension. Please call `as_tensor()` "
                         f"beforehand")

    return tensor._from_data(
        tensor._map,
        aten.unsqueeze.default(tensor.as_tensor(), dim),
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


@implements(aten.squeeze.default)
def _squeeze_default(tensor: HashTensor) -> HashTensor:
    if tensor._value is None:
        return tensor._shallow_copy()

    value = tensor.as_tensor()
    for d in range(tensor.dim() - 1, 0, -1):
        value = value.squeeze(d)

    return tensor._from_data(
        tensor._map,
        value,
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


@implements(aten.squeeze.dim)
@implements(getattr(aten.squeeze, 'dims', aten.squeeze.dim))
def _squeeze_dim(
    tensor: HashTensor,
    dim: Union[int, List[int]],
) -> HashTensor:
    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d < -tensor.dim() or d >= tensor.dim():
            raise IndexError(f"Dimension out of range (expected to be in "
                             f"range of [{-tensor.dim()}, {tensor.dim()-1}], "
                             f"but got {d})")

    if tensor._value is None:
        return tensor._shallow_copy()

    value = tensor.as_tensor()
    for d in dim[::-1]:
        if d != 0 and d != -tensor.dim():
            value = value.squeeze(d)

    return tensor._from_data(
        tensor._map,
        value,
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


@implements(aten.slice.Tensor)
def _slice(
    tensor: HashTensor,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> HashTensor:

    if dim == 0 or dim == -tensor.dim():
        copy = start is None or start == 0 or start <= -tensor.size(0)
        copy &= end is None or end > tensor.size(0)
        copy &= step == 1
        if copy:
            return tensor._shallow_copy()

        key = aten.slice.Tensor(tensor._key, 0, start, end, step)
        value = aten.slice.Tensor(tensor.as_tensor(), 0, start, end, step)
        return tensor.__class__(key, value)

    return tensor._from_data(
        tensor._map,
        aten.slice.Tensor(tensor.as_tensor(), dim, start, end, step),
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


# Since PyTorch does only allow PyTorch tensors as indices in `index_select`,
# we need to create a wrapper function and monkey patch `index_select` :(
_old_index_select = torch.index_select


def _new_index_select(
    input: Tensor,
    dim: Union[int, str],
    index: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:

    if isinstance(dim, int) and (dim < -input.dim() or dim >= input.dim()):
        raise IndexError(f"Dimension out of range (expected to be in range of "
                         f"[{-input.dim()}, {input.dim()-1}], but got {dim})")

    # We convert any index tensor in the first dimension into a tensor. This
    # means that downstream handling (i.e. in `aten.index_select.default`)
    # needs to take this pre-conversion into account.
    if (not torch.jit.is_scripting() and isinstance(input, HashTensor)
            and isinstance(dim, int) and (dim == 0 or dim == -input.dim())):
        index = as_key_tensor(index, device=input.device)

    if isinstance(dim, int):  # Type narrowing...
        if out is None:
            return _old_index_select(input, dim, index)
        else:
            return _old_index_select(input, dim, index, out=out)
    else:
        if out is None:
            return _old_index_select(input, dim, index)
        else:
            return _old_index_select(input, dim, index, out=out)


torch.index_select = _new_index_select  # type: ignore


@implements(aten.index_select.default)
def _index_select(
    tensor: HashTensor,
    dim: int,
    index: Tensor,
) -> Union[HashTensor, Tensor]:

    if dim == 0 or dim == -tensor.dim():
        return tensor._get(index)

    return tensor._from_data(
        tensor._map,
        aten.index_select.default(tensor.as_tensor(), dim, index),
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


# Since PyTorch does only allow PyTorch tensors as indices in `select`, we need
# to create a wrapper function and monkey patch `select` :(
_old_select = torch.select


def _new_select(
    input: Tensor,
    dim: Union[int, str],
    index: int,
) -> Tensor:

    if isinstance(dim, int) and (dim < -input.dim() or dim >= input.dim()):
        raise IndexError(f"Dimension out of range (expected to be in range of "
                         f"[{-input.dim()}, {input.dim()-1}], but got {dim})")

    # We convert any index in the first dimension into an integer. This means
    # that downstream handling (i.e. in `aten.select.int`) needs to take this
    # pre-conversion into account.
    if (not torch.jit.is_scripting() and isinstance(input, HashTensor)
            and isinstance(dim, int) and (dim == 0 or dim == -input.dim())):
        index = int(as_key_tensor([index]))

    if isinstance(dim, int):  # Type narrowing...
        return _old_select(input, dim, index)
    else:
        return _old_select(input, dim, index)


torch.select = _new_select  # type: ignore


@implements(aten.select.int)
def _select(
    tensor: HashTensor,
    dim: int,
    index: int,
) -> Union[HashTensor, Tensor]:

    if dim == 0 or dim == -tensor.dim():
        key = torch.tensor(
            [index],
            dtype=tensor._min_key.dtype,
            device=tensor.device,
        )
        return tensor._get(key).squeeze(0)

    return tensor._from_data(
        tensor._map,
        aten.select.int(tensor.as_tensor(), dim, index),
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


@implements(aten.index.Tensor)
def _index(
    tensor: HashTensor,
    indices: List[Optional[Tensor]],
) -> Union[HashTensor, Tensor]:

    assert len(indices) > 0

    if indices[0] is not None:
        out = tensor._get(indices[0])
        if len(indices) > 1:
            out = aten.index.Tensor(out, [None] + indices[1:])
        return out

    return tensor._from_data(
        tensor._map,
        aten.index.Tensor(tensor.as_tensor(), indices),
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )
