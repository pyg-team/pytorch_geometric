from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import torch
import torch.utils._pytree as pytree
import xxhash
from torch import Tensor

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
        key = torch.tensor([
            xxhash.xxh64(item).intdigest() & 0x7FFFFFFFFFFFFFFF for item in key
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
