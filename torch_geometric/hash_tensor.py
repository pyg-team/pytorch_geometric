from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor

from torch_geometric.typing import CPUHashMap, CUDAHashMap

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def to_key_tensor(
    key: Any,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    if not isinstance(key, Tensor):
        key = torch.tensor(key, device=device)
    else:
        key = key.to(device)
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
            if not isinstance(value, Tensor):
                value = torch.tensor(value, dtype=dtype, device=device)
            else:
                value = value.to(device, dtype)

        device = value.device if value is not None else device
        key = to_key_tensor(key, device=device)
        device = key.device

        # TODO Add numpy support
        if not isinstance(key, Tensor):
            device = value.device if value is not None else device
            key = torch.tensor(key, device=device)
        else:
            key = key.to(device)
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
