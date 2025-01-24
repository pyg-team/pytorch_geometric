import gc
import os
import os.path as osp
import random
import subprocess as sp
import sys
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Tuple

import torch
from torch import Tensor

from torch_geometric.data.data import BaseData
from torch_geometric.typing import SparseTensor


def count_parameters(model: torch.nn.Module) -> int:
    r"""Given a :class:`torch.nn.Module`, count its trainable parameters.

    Args:
        model (torch.nn.Model): The model.
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_model_size(model: torch.nn.Module) -> int:
    r"""Given a :class:`torch.nn.Module`, get its actual disk size in bytes.

    Args:
        model (torch model): The model.
    """
    path = f'{random.randrange(sys.maxsize)}.pt'
    torch.save(model.state_dict(), path)
    model_size = osp.getsize(path)
    os.remove(path)
    return model_size


def get_data_size(data: BaseData) -> int:
    r"""Given a :class:`torch_geometric.data.Data` object, get its theoretical
    memory usage in bytes.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
    """
    data_ptrs = set()

    def _get_size(obj: Any) -> int:
        if isinstance(obj, Tensor):
            if obj.data_ptr() in data_ptrs:
                return 0
            data_ptrs.add(obj.data_ptr())
            return obj.numel() * obj.element_size()
        elif isinstance(obj, SparseTensor):
            return _get_size(obj.csr())
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            return sum([_get_size(x) for x in obj])
        elif isinstance(obj, Mapping):
            return sum([_get_size(x) for x in obj.values()])
        else:
            return 0

    return sum([_get_size(store) for store in data.stores])


def get_cpu_memory_from_gc() -> int:
    r"""Returns the used CPU memory in bytes, as reported by the
    :python:`Python` garbage collector.
    """
    warnings.filterwarnings('ignore', '.*torch.distributed.reduce_op.*')

    mem = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Tensor) and not obj.is_cuda:
                mem += obj.numel() * obj.element_size()
        except Exception:
            pass
    return mem


def get_gpu_memory_from_gc(device: int = 0) -> int:  # pragma: no cover
    r"""Returns the used GPU memory in bytes, as reported by the
    :python:`Python` garbage collector.

    Args:
        device (int, optional): The GPU device identifier. (default: :obj:`1`)
    """
    warnings.filterwarnings('ignore', '.*torch.distributed.reduce_op.*')

    mem = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Tensor) and obj.get_device() == device:
                mem += obj.numel() * obj.element_size()
        except Exception:
            pass
    return mem


def get_gpu_memory_from_nvidia_smi(  # pragma: no cover
    device: int = 0,
    digits: int = 2,
) -> Tuple[float, float]:
    r"""Returns the free and used GPU memory in megabytes, as reported by
    :obj:`nivdia-smi`.

    .. note::

        :obj:`nvidia-smi` will generally overestimate the amount of memory used
        by the actual program, see `here <https://pytorch.org/docs/stable/
        notes/faq.html#my-gpu-memory-isn-t-freed-properly>`__.

    Args:
        device (int, optional): The GPU device identifier. (default: :obj:`1`)
        digits (int): The number of decimals to use for megabytes.
            (default: :obj:`2`)
    """
    CMD = 'nvidia-smi --query-gpu=memory.free --format=csv'
    free_out = sp.check_output(CMD.split()).decode('utf-8').split('\n')[1:-1]

    CMD = 'nvidia-smi --query-gpu=memory.used --format=csv'
    used_out = sp.check_output(CMD.split()).decode('utf-8').split('\n')[1:-1]

    if device < 0 or device >= len(free_out):
        raise AttributeError(
            f'GPU {device} not available (found {len(free_out)} GPUs)')

    free_mem = medibyte_to_megabyte(int(free_out[device].split()[0]), digits)
    used_mem = medibyte_to_megabyte(int(used_out[device].split()[0]), digits)

    return free_mem, used_mem


def get_gpu_memory_from_ipex(
        device: int = 0,
        digits=2) -> Tuple[float, float, float]:  # pragma: no cover
    r"""Returns the XPU memory statistics.

    Args:
        device (int, optional): The GPU device identifier. (default: :obj:`0`)
        digits (int): The number of decimals to use for megabytes.
            (default: :obj:`2`)
    """
    import intel_extension_for_pytorch as ipex
    stats = ipex.xpu.memory_stats_as_nested_dict(device)
    max_allocated = stats['allocated_bytes']['all']['peak']
    max_reserved = stats['reserved_bytes']['all']['peak']
    max_active = stats['active_bytes']['all']['peak']
    max_allocated = byte_to_megabyte(max_allocated, digits)
    max_reserved = byte_to_megabyte(max_reserved, digits)
    max_active = byte_to_megabyte(max_active, digits)
    ipex.xpu.reset_peak_memory_stats(device)
    return max_allocated, max_reserved, max_active


###############################################################################


def byte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(value / (1024 * 1024), digits)


def medibyte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(1.0485 * value, digits)
