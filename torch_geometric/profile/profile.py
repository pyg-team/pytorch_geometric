import os
import pathlib
import time
from contextlib import ContextDecorator, contextmanager
from typing import Any, List, NamedTuple, Tuple

import torch
from torch.autograd.profiler import EventList
from torch.profiler import ProfilerActivity, profile

from torch_geometric.profile.utils import (
    byte_to_megabyte,
    get_gpu_memory_from_nvidia_smi,
)


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


def profileit():  # pragma: no cover
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

            device = None
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, torch.Tensor):
                    device = arg.get_device()
                    break
            if device is None:
                raise AttributeError(
                    "Could not infer CUDA device from the args in the "
                    "function being profiled")
            if device == -1:
                raise RuntimeError(
                    "The profiling decorator does not support profiling "
                    "on non CUDA devices")

            # Init `pytorch_memlab` for analyzing the model forward pass:
            line_profiler = LineProfiler(target_gpu=device)
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
            free_cuda, used_cuda = get_gpu_memory_from_nvidia_smi(
                device=device)

            stats = Stats(time, max_allocated_cuda, max_reserved_cuda,
                          max_active_cuda, free_cuda, used_cuda)

            return out, stats

        return wrapper

    return decorator


class timeit(ContextDecorator):
    r"""A context decorator to facilitate timing a function, *e.g.*, obtaining
    the runtime of a specific model on a specific dataset.

    .. code-block:: python

        @torch.no_grad()
        def test(model, x, edge_index):
            return model(x, edge_index)

        with timeit() as t:
            z = test(model, x, edge_index)
        time = t.duration

    Args:
        log (bool, optional): If set to :obj:`False`, will not log any runtime
            to the console. (default: :obj:`True`)
        avg_time_divisor (int, optional): If set to a value greater than
            :obj:`1`, will divide the total time by this value. Useful for
            calculating the average of runtimes within a for-loop.
            (default: :obj:`0`)
    """
    def __init__(self, log: bool = True, avg_time_divisor: int = 0):
        self.log = log
        self.avg_time_divisor = avg_time_divisor

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t_start = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t_end = time.time()
        self.duration = self.t_end - self.t_start
        if self.avg_time_divisor > 1:
            self.duration = self.duration / self.avg_time_divisor
        if self.log:  # pragma: no cover
            print(f'Time: {self.duration:.8f}s', flush=True)

    def reset(self):
        r"""Prints the duration and resets current timer."""
        if self.t_start is None:
            raise RuntimeError("Timer wasn't started.")
        else:
            self.__exit__()
            self.__enter__()


def get_stats_summary(stats_list: List[Stats]):  # pragma: no cover
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
        time_mean=float(torch.tensor([s.time for s in stats_list]).mean()),
        time_std=float(torch.tensor([s.time for s in stats_list]).std()),
        max_allocated_cuda=max([s.max_allocated_cuda for s in stats_list]),
        max_reserved_cuda=max([s.max_reserved_cuda for s in stats_list]),
        max_active_cuda=max([s.max_active_cuda for s in stats_list]),
        min_nvidia_smi_free_cuda=min(
            [s.nvidia_smi_free_cuda for s in stats_list]),
        max_nvidia_smi_used_cuda=max(
            [s.nvidia_smi_used_cuda for s in stats_list]),
    )


###############################################################################


def read_from_memlab(line_profiler: Any) -> List[float]:  # pragma: no cover
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


def trace_handler(p):
    print_time_total(p)
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'timeline' + '.json'
    p.export_chrome_trace(timeline_file)


def print_time_total(p):
    if torch.cuda.is_available():
        profile_sort = 'self_cuda_time_total'
    else:
        profile_sort = 'self_cpu_time_total'
    output = p.key_averages().table(sort_by=profile_sort)
    print(output)


def rename_profile_file(*args):
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'profile'
    for arg in args:
        timeline_file += '-' + arg
    timeline_file += '.json'
    os.rename('timeline.json', timeline_file)


@contextmanager
def torch_profile(export_chrome_trace=True, csv_data=None, write_csv=None):
    use_cuda = torch.cuda.is_available()

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    if export_chrome_trace:
        p_trace_handler = trace_handler
    else:
        p_trace_handler = print_time_total

    p = profile(activities=activities, on_trace_ready=p_trace_handler)

    with p:
        yield
        p.step()

    if csv_data is not None and write_csv == 'prof':
        if use_cuda:
            profile_sort = 'self_cuda_time_total'
        else:
            profile_sort = 'self_cpu_time_total'
        events = EventList(
            sorted(
                p.key_averages(),
                key=lambda evt: getattr(evt, profile_sort),
                reverse=True,
            ), use_cuda=use_cuda)

        save_profile_data(csv_data, events, use_cuda)


@contextmanager
def xpu_profile(export_chrome_trace=True):
    with torch.autograd.profiler_legacy.profile(use_xpu=True) as profile:
        yield
    print(profile.key_averages().table(sort_by='self_xpu_time_total'))
    if export_chrome_trace:
        profile.export_chrome_trace('timeline.json')


def format_prof_time(time):
    # Profile time is in micro seconds, so format it appropriately:
    return round(time / 1e6, 3)


def save_profile_data(csv_data, events, use_cuda):
    sum_self_cpu_time_total = sum(
        [event.self_cpu_time_total for event in events])
    sum_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    sum_self_cuda_time_total = sum(
        [event.self_cuda_time_total for event in events]) if use_cuda else 0

    for e in events[:5]:  # Save top 5 most time consuming operations:
        csv_data['NAME'].append(e.key)
        csv_data['SELF CPU %'].append(
            round(e.self_cpu_time_total * 100.0 / sum_self_cpu_time_total, 3))
        csv_data['SELF CPU'].append(format_prof_time(e.self_cpu_time_total))
        csv_data['CPU TOTAL %'].append(
            round(e.cpu_time_total * 100.0 / sum_cpu_time_total, 3))
        csv_data['CPU TOTAL'].append(format_prof_time(e.cpu_time_total))
        csv_data['CPU TIME AVG'].append(format_prof_time(e.cpu_time_total))
        if use_cuda:
            csv_data['SELF CUDA %'].append(e.self_cuda_time_total * 100.0 /
                                           sum_self_cuda_time_total)
            csv_data['SELF CUDA'].append(
                format_prof_time(e.self_cuda_time_total))
            csv_data['CUDA TOTAL'].append(format_prof_time(e.cpu_time_total))
            csv_data['CUDA TIME AVG'].append(format_prof_time(
                e.cpu_time_total))
        csv_data['# OF CALLS'].append(e.count)
