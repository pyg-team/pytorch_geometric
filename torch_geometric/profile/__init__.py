r"""GNN profiling package."""

from .benchmark import benchmark
from .profile import (
    get_stats_summary,
    print_time_total,
    profileit,
    rename_profile_file,
    timeit,
    torch_profile,
    trace_handler,
    xpu_profile,
)
from .utils import (
    count_parameters,
    get_cpu_memory_from_gc,
    get_data_size,
    get_gpu_memory_from_gc,
    get_gpu_memory_from_ipex,
    get_gpu_memory_from_nvidia_smi,
    get_model_size,
)
from .nvtx import nvtxit

__all__ = [
    'profileit',
    'timeit',
    'get_stats_summary',
    'trace_handler',
    'print_time_total',
    'rename_profile_file',
    'torch_profile',
    'xpu_profile',
    'count_parameters',
    'get_model_size',
    'get_data_size',
    'get_cpu_memory_from_gc',
    'get_gpu_memory_from_gc',
    'get_gpu_memory_from_nvidia_smi',
    'get_gpu_memory_from_ipex',
    'benchmark',
    'nvtxit',
]

classes = __all__
