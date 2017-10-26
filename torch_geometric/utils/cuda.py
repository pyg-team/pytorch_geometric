from collections import namedtuple
from string import Template

import torch

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


if torch.cuda.is_available():
    import cupy

    @cupy.util.memoize(for_each_device=True)
    def load_kernel(kernel_name, code, **kwargs):
        code = Template(code).substitute(**kwargs)
        kernel_code = cupy.cuda.compile_with_cache(code)
        return kernel_code.get_function(kernel_name)
