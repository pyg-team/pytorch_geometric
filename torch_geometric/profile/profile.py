from typing import Any, List, NamedTuple, Tuple

import torch

from torch_geometric.profile.utils import (byte_to_megabyte,
                                           get_gpu_memory_from_nvidia_smi)


class Stats(NamedTuple):
    time: float
    max_allocated_cuda: float
    max_reserved_cuda: float
    max_active_cuda: float
    nvidia_smi_free_cuda: float
    nvidia_smi_used_cuda: float


class StatsSummary(NamedTuple):
    time_mean: float
    time_std: float
    max_allocated_cuda: float
    max_reserved_cuda: float
    max_active_cuda: float
    min_nvidia_smi_free_cuda: float
    max_nvidia_smi_used_cuda: float


def profileit():
    r"""A decorator to facilitate profiling a function, *e.g.*, obtaining
    training runtime and memory statistics of a specific model on a specific
    dataset.
    Returns a :obj:`Stats` object with the attributes :obj:`time`,
    :obj:`max_active_cuda`, :obj:`max_reserved_cuda`, :obj:`max_active_cuda`,
    :obj:`nvidia_smi_free_cuda`, :obj:`nvidia_smi_used_cuda`.

    .. code-block:: python

        @profileit()
        def train(model, optimizer, x, edge_index, y):
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            return float(loss)

        loss, stats = train(model, x, edge_index, y)
    """
    def decorator(func):
        def wrapper(*args, **kwargs) -> Tuple[Any, Stats]:
            from pytorch_memlab import LineProfiler

            model = args[0]
            if not isinstance(model, torch.nn.Module):
                raise AttributeError(
                    'First argument for profiling needs to be torch.nn.Module')

            # Init `pytorch_memlab` for analyzing the model forward pass:
            line_profiler = LineProfiler()
            line_profiler.enable()
            line_profiler.add_function(args[0].forward)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            out = func(*args, **kwargs)

            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000

            # Get the global memory statistics collected by `pytorch_memlab`:
            memlab = read_from_memlab(line_profiler)
            max_allocated_cuda, max_reserved_cuda, max_active_cuda = memlab
            line_profiler.disable()

            # Get additional information from `nvidia-smi`:
            free_cuda, used_cuda = get_gpu_memory_from_nvidia_smi()

            stats = Stats(time, max_allocated_cuda, max_reserved_cuda,
                          max_active_cuda, free_cuda, used_cuda)

            return out, stats

        return wrapper

    return decorator


def timeit():
    r"""A decorator to facilitate timing a function, *e.g.*, obtaining the
    runtime of a specific model on a specific dataset.

    .. code-block:: python

        @timeit()
        @torch.no_grad()
        def test(model, x, edge_index):
            return model(x, edge_index)

        z, time = test(model, x, edge_index)
    """
    def decorator(func):
        def wrapper(*args, **kwargs) -> Tuple[Any, float]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            out = func(*args, **kwargs)

            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000

            return out, time

        return wrapper

    return decorator


def get_stats_summary(stats_list: List[Stats]):
    r"""Creates a summary of collected runtime and memory statistics.
    Returns a :obj:`StatsSummary` object with the attributes :obj:`time_mean`,
    :obj:`time_std`,
    :obj:`max_active_cuda`, :obj:`max_reserved_cuda`, :obj:`max_active_cuda`,
    :obj:`min_nvidia_smi_free_cuda`, :obj:`max_nvidia_smi_used_cuda`.

    Args:
        stats_list (List[Stats]): A list of :obj:`Stats` objects, as returned
            by :meth:`~torch_geometric.profile.profileit`.
    """
    return StatsSummary(
        time_mean=mean([stats.time for stats in stats_list]),
        time_std=std([stats.time for stats in stats_list]),
        max_allocated_cuda=max([s.max_allocated_cuda for s in stats_list]),
        max_reserved_cuda=max([s.max_reserved_cuda for s in stats_list]),
        max_active_cuda=max([s.max_active_cuda for s in stats_list]),
        min_nvidia_smi_free_cuda=min(
            [s.nvidia_smi_free_cuda for s in stats_list]),
        max_nvidia_smi_used_cuda=max(
            [s.nvidia_smi_used_cuda for s in stats_list]),
    )


###############################################################################


def read_from_memlab(line_profiler: Any) -> List[float]:
    from pytorch_memlab.line_profiler.line_records import LineRecords

    # See: https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

    track_stats = [  # Different statistic can be collected as needed.
        'allocated_bytes.all.peak',
        'reserved_bytes.all.peak',
        'active_bytes.all.peak',
    ]

    records = LineRecords(line_profiler._raw_line_records,
                          line_profiler._code_infos)
    stats = records.display(None, track_stats)._line_records
    return [byte_to_megabyte(x) for x in stats.values.max(axis=0).tolist()]


def std(values: List[float]):
    return float(torch.tensor(values).std())


def mean(values: List[float]):
    return float(torch.tensor(values).mean())
