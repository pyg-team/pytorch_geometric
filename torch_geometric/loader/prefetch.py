from contextlib import nullcontext
from functools import partial
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from torch_geometric.typing import WITH_IPEX


class PrefetchLoader:
    r"""A GPU prefetcher class for asynchronously transferring data of a
    :class:`torch.utils.data.DataLoader` from host memory to device memory.

    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device, optional): The device to load the data to.
            (default: :obj:`None`)
    """
    class PrefetchLoaderDevice:
        def __init__(self, device: Optional[torch.device] = None):
            cuda_present = torch.cuda.is_available()
            xpu_present = torch.xpu.is_available() if WITH_IPEX else False

            if device is None:
                if cuda_present:
                    device = 'cuda'
                elif xpu_present:
                    device = 'xpu'
                else:
                    device = 'cpu'

            self.device = torch.device(device)

            if ((self.device.type == 'cuda' and not cuda_present)
                    or (self.device.type == 'xpu' and not xpu_present)):
                print(f'Requested device[{self.device.type}] is not available '
                      '- fallback to CPU')
                self.device = torch.device('cpu')

            self.is_gpu = self.device.type in ['cuda', 'xpu']
            self.stream = None
            self.stream_context = nullcontext

            if self.is_gpu:
                gpu_module = torch.cuda if self.device.type == 'cuda' else torch.xpu
            else:
                gpu_module = None

            self.gpu_module = gpu_module

        def maybe_init_stream(self) -> None:
            if self.is_gpu:
                self.stream = self.gpu_module.Stream()
                self.stream_context = partial(self.gpu_module.stream,
                                              stream=self.stream)

        def maybe_wait_stream(self) -> None:
            if self.stream is not None:
                self.gpu_module.current_stream().wait_stream(self.stream)

        def get_device(self) -> torch.device:
            return self.device

        def get_stream_context(self) -> Any:
            return self.stream_context()

    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        self.loader = loader
        self.device_mgr = self.PrefetchLoaderDevice(device)

    def non_blocking_transfer(self, batch: Any) -> Any:
        if not self.device_mgr.is_gpu:
            return batch
        if isinstance(batch, (list, tuple)):
            return [self.non_blocking_transfer(v) for v in batch]
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        device = self.device_mgr.get_device()
        batch = batch.pin_memory(device)
        return batch.to(device, non_blocking=True)

    def __iter__(self) -> Any:
        first = True
        self.device_mgr.maybe_init_stream()

        for next_batch in self.loader:

            with self.device_mgr.get_stream_context():
                next_batch = self.non_blocking_transfer(next_batch)

            if not first:
                yield batch  # noqa
            else:
                first = False

            self.device_mgr.maybe_wait_stream()

            batch = next_batch

        yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
