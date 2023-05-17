from contextlib import nullcontext
from functools import partial
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


class PrefetchLoader:
    r"""A GPU prefetcher class for asynchronously transferring data of a
    :class:`torch.utils.data.DataLoader` from host memory to device memory.

    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device, optional): The device to load the data to.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.loader = loader
        self.device = torch.device(device)

        self.is_cuda = torch.cuda.is_available() and self.device.type == 'cuda'

    def non_blocking_transfer(self, batch: Any) -> Any:
        if not self.is_cuda:
            return batch
        if isinstance(batch, (list, tuple)):
            return [self.non_blocking_transfer(v) for v in batch]
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        batch = batch.pin_memory()
        return batch.to(self.device, non_blocking=True)

    def __iter__(self) -> Any:
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = nullcontext

        for next_batch in self.loader:

            with stream_context():
                next_batch = self.non_blocking_transfer(next_batch)

            if not first:
                yield batch  # noqa
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            batch = next_batch

        yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
