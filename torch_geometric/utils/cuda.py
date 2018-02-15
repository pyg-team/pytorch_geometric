from __future__ import division

from collections import namedtuple
from string import Template

import torch

cuda_num_threads = 512
Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                        \
      i += blockDim.x * gridDim.x)
'''


def get_blocks(n, k=cuda_num_threads):
    return (n + k - 1) // k


if torch.cuda.is_available():
    import cupy

    @cupy.util.memoize(for_each_device=True)
    def load_kernel(kernel_name, code, **kwargs):
        code = Template(code).substitute(**kwargs)
        kernel_code = cupy.cuda.compile_with_cache(code)
        return kernel_code.get_function(kernel_name)
