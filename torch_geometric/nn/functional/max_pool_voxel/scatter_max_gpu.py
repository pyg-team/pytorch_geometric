import torch

from ....utils.cuda import (cuda_num_threads, Stream, load_kernel, kernel_loop,
                            get_blocks)
from .int_as_float_gpu import int_as_float_gpu

kernel = kernel_loop + """
extern "C"
__global__ void scatter_max(
const float* input, const int* cluster, int* max) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    const int n_idx = idx / ${M};
    const int m_idx = idx % ${M};
    int c = cluster[n_idx] * ${M} + m_idx;

    // Convert value `f` to ordered integer value to perform `atomicMax`.
    int f = __float_as_int(input[idx]);
    f = f >= 0 ? f : f ^ 0x7FFFFFFF;

    // Compute `scatter_max`.
    atomicMax((int*) &max[c], f);
  }
}
"""


def scatter_max_gpu(input, cluster, K):
    with torch.cuda.device_of(input):
        M = input.size(1)
        max = torch.cuda.IntTensor(K, M).fill_(-2147483648)
        num_threads = input.numel()
        f = load_kernel('scatter_max', kernel, num_threads=num_threads, M=M)
        f(args=[input.data_ptr(),
                cluster.data_ptr(),
                max.data_ptr()],
          block=(cuda_num_threads, 1, 1),
          grid=(get_blocks(num_threads), 1, 1),
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return int_as_float_gpu(max)
