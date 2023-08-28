from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader


def to_device(inputs: Any, device: Optional[torch.device] = None) -> Any:
    if hasattr(inputs, 'to'):
        return inputs.to(device)
    elif isinstance(inputs, Mapping):
        return {key: to_device(value, device) for key, value in inputs.items()}
    elif isinstance(inputs, tuple) and hasattr(inputs, '_fields'):
        return type(inputs)(*(to_device(s, device) for s in zip(*inputs)))
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        return [to_device(s, device) for s in zip(*inputs)]

    return inputs


class CachedLoader:
    r"""A loader to cache mini-batch outputs, e.g., obtained during
    :class:`NeighborLoader` iterations.

    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device, optional): The device to load the data to.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
        transform: Optional[Callable] = None,
    ):
        self.loader = loader
        self.device = device
        self.transform = transform

        self._cache: List[Any] = []

    def clear(self):
        r"""Clears the cache."""
        self._cache = []

    def __iter__(self) -> Any:
        if len(self._cache):
            for batch in self._cache:
                yield batch
            return

        for batch in self.loader:

            if self.transform is not None:
                batch = self.transform(batch)

            batch = to_device(batch, self.device)

            self._cache.append(batch)

            yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
