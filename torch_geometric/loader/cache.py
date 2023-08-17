from typing import Any, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

BatchType = List[Tuple[Tensor, Tensor, int]]


class CachedLoader:
    r"""A caching loader intended to use in layer-wise inference
    performed on GPU. During the first walk through the data
    stored in :class:`torch.utils.data.DataLoader` batches are
    transferred to GPU memory and cached.

    Args:
        loader (torch.utils.data.DataLoader): The wrapped data loader.
        device (torch.device): The device to load the data to.
    """
    def __init__(self, loader: DataLoader, device: torch.device):
        self.n_batches = len(loader)
        self.iter = iter(loader)
        self.cache_filled = False
        self.device = device
        self.cache = []

    def cache_batch(self, batch: Any) -> BatchType:
        n_id = batch.n_id.to(self.device)
        if hasattr(batch, 'adj_t'):
            edge_index = batch.adj_t.to(self.device)
        else:
            edge_index = batch.edge_index.to(self.device)
        batch = (n_id, edge_index, batch.batch_size)
        self.cache.append(batch)
        return batch

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> BatchType:
        try:
            batch = next(self.iter)
            if not self.cache_filled:
                batch = self.cache_batch(batch)
            return batch
        except StopIteration as exc:
            self.cache_filled = True
            self.iter = iter(self.cache)
            raise StopIteration from exc

    def __len__(self) -> int:
        return self.n_batches

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
