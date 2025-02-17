import functools
import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

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
        # TODO Convert int64 to int32.
        # On GPU, we default to int32 for faster 'CUDAHashMap' implementation:
        if torch_geometric.typing.WITH_CUDA_HASH_MAP and device.type == 'cuda':
            pass
            # key = torch.tensor(
            #     [xxhash.xxh32(x).intdigest() & 0x7FFFFFFF for x in key],
            #     dtype=torch.int32, device=device)
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
        # TODO Convert int64 to int32.
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
                size=(_range + 2, ),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            )
            _map[(key - (min_key - 1)).long()] = torch.arange(
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

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`HashTensor` representation back to a
        :class:`torch.Tensor` representation.
        """
        if self._value is not None:
            return self._value
        return torch.arange(self.size(0), dtype=self.dtype, device=self.device)

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
    elif (torch_geometric.typing.WITH_CUDA_HASH_MAP and min_key.is_cuda
          and tensor._min_key.device != min_key.device):
        key = _map.keys()
        key = aten._to_copy.default(key, device=device)
        _map = get_hash_map(key)

    return tensor.__class__._from_data(
        _map,
        value,
        min_key,
        max_key,
        num_keys=tensor.size(0),
        dtype=dtype or tensor.dtype,
    )
