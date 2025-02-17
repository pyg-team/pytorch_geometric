import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import torch
import torch.utils._pytree as pytree
import xxhash
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import CPUHashMap, CUDAHashMap

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def as_key_tensor(
    key: Any,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    try:
        key = torch.as_tensor(key, device=device)
    except Exception:
        device = device or torch.get_default_device()
        # On GPU, we default to int32 for faster 'CUDAHashMap' implementation:
        if device.type == 'cuda':
            key = torch.tensor(
                [xxhash.xxh32(x).intdigest() & 0x7FFFFFFF for x in key],
                dtype=torch.int32, device=device)
        else:
            key = torch.tensor([
                xxhash.xxh64(x).intdigest() & 0x7FFFFFFFFFFFFFFF for x in key
            ], dtype=torch.int64, device=device)

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
        device = key.device

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

            dtype = value.dtype
            size = value.size()
            stride = value.stride()
            layout = value.layout
            requires_grad = value.requires_grad
        else:
            dtype = dtype or torch.int64
            size = torch.Size([key.numel()])
            stride = (1, )
            layout = torch.strided
            requires_grad = False

        out = Tensor._make_wrapper_subclass(  # type: ignore
            cls,
            size=size,
            strides=stride,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=requires_grad,
        )

        out._value = value

        out._min_key = key.min() if key.numel() > 0 else key.new_zeros(())
        out._max_key = key.max() if key.numel() > 0 else key.new_zeros(())

        _range = out._max_key - out._min_key
        # TODO Expose fixed threshold as argument.
        if (key.dtype in {torch.uint8, torch.int16} or _range <= 1_000_000
                or _range <= 2 * key.numel()):
            out._map = torch.full(
                size=(_range + 2, ),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            )
            out._map[(key - (out._min_key - 1)).long()] = torch.arange(
                key.numel(),
                dtype=out._map.dtype,
                device=out._map.device,
            )
        elif torch_geometric.typing.WITH_CUDA_HASH_MAP and key.is_cuda:
            # TODO Convert int64 to int32.
            out._map = CUDAHashMap(key, 0.5)
        elif torch_geometric.typing.WITH_CPU_HASH_MAP and key.is_cpu:
            out._map = CPUHashMap(key, -1)
        else:
            if key.is_cuda:
                warnings.warn(f"Fallback to CPU-based mapping algorithm which "
                              f"may cause slowdowns and device "
                              f"synchronization. Please install 'pyg-lib' for "
                              f"an accelerated '{cls.__name__}' "
                              f"implementation.")

            import pandas as pd

            out._map = pd.CategoricalDtype(
                categories=key.cpu().numpy(),
                ordered=True,
            )

        return out

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`HashTensor` representation back to a
        :class:`torch.Tensor` representation.
        """
        if self._value is not None:
            return self._value
        return torch.arange(self.size(0), dtype=self.dtype, device=self.device)

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
