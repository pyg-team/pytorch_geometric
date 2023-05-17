from queue import Queue
from threading import Thread
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


class GPUPrefetcher:
    r"""A GPU prefetcher class for asynchronously loading data from a
    :class:`torch.utils.data.DataLoader` from host memory to device memory.

    Args:
        loader (torch.utils.DataLoader): A data loader object.
        device (torch.device): The CUDA device to load the data to.
        prefetch_size (int, optional): The number of batches to prefetch at
            once. (default: :obj:`1`)
    """
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        prefetch_size: int = 1,
    ):
        if prefetch_size < 1:
            raise ValueError(f"'prefetch_size' must be greater than 0 "
                             f"(got {prefetch_size})")

        self.loader = loader
        self.device = torch.device(device)
        self.prefetch_size = prefetch_size

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=prefetch_size)
        self.worker: Optional[Thread] = None

        self.idx = 0

    def non_blocking_transfer(self, batch: Any) -> Any:
        # (Recursive) non-blocking device transfer:
        if isinstance(batch, (list, tuple)):
            return [self.non_blocking_transfer(v) for v in batch]
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        with torch.cuda.stream(self.load_stream):
            if not batch.is_pinned():
                batch = batch.pin_memory()
            return batch.to(self.device, non_blocking=True)

    def load_loop(self):
        for batch in self.loader:
            self.queue.put(self.non_blocking_transfer(batch))

    def __iter__(self) -> 'GPUPrefetcher':
        is_dead = self.worker is None or not self.worker.is_alive()
        if is_dead and self.queue.empty() and self.idx == 0:
            self.worker = Thread(target=self.load_loop)
            self.worker.daemon = True
            self.worker.start()

        return self

    def __next__(self) -> Any:
        is_dead = not self.worker.is_alive()
        if (is_dead and self.queue.empty()) or self.idx >= len(self):
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration

        out = self.queue.get()
        self.queue.task_done()
        self.idx += 1
        return out

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
