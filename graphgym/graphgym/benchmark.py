# Do not change; required for benchmarking
import time
import random

import torch_geometric

import torch_geometric_benchmark.torchprof_local as torchprof
from torch_geometric_benchmark.utils import get_max_cuda
from torch_geometric_benchmark.utils import get_logger
from torch_geometric_benchmark.utils import layer_trace
from torch_geometric_benchmark.utils import get_memory_status
from torch_geometric_benchmark.utils import count_parameters
from torch_geometric_benchmark.utils import get_gpu_memory_nvdia
from torch_geometric_benchmark.utils import get_model_size
from torch_geometric_benchmark.utils import get_summary
from pytorch_memlab import LineProfiler


global_line_profiler = LineProfiler()
global_line_profiler.enable()