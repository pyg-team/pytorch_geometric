import torch

from ....utils.cuda import (cuda_num_threads, Stream, Dtype, load_kernel,
                            kernel_loop, get_blocks)

kernel = kernel_loop + """
extern "C"
__global__ void scatter_arg_max(
const ${Dtype}* input, const long* cluster, const ${Dtype}* max,
long* argmax) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    const int n_idx = idx / ${M};
    const int m_idx = idx % ${M};

    int c = cluster[n_idx] * ${M} + m_idx;

    ${Dtype} f = input[idx];
    ${Dtype} m = max[c];

    if (f == m) argmax[c] = n_idx;
  }
}
"""


def scatter_arg_max_gpu(input, cluster, max):
    with torch.cuda.device_of(input):
        K, M = max.size()
        argmax = torch.cuda.LongTensor(K, M)
        num_threads = input.numel()
        f = load_kernel(
            'scatter_arg_max',
            kernel,
            Dtype=Dtype(input),
            num_threads=num_threads,
            M=M)
        f(args=[
            input.data_ptr(),
            cluster.data_ptr(),
            max.data_ptr(),
            argmax.data_ptr()
        ],
          block=(cuda_num_threads, 1, 1),
          grid=(get_blocks(num_threads), 1, 1),
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return argmax
