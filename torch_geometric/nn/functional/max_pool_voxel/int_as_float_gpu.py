import torch

from ....utils.cuda import (cuda_num_threads, Stream, load_kernel, kernel_loop,
                            get_blocks)

kernel = kernel_loop + """
extern "C"
__global__ void int_as_float(const int* input, float* output) {
  CUDA_KERNEL_LOOP(idx, ${num_threads}) {
    int v = input[idx];
    output[idx] = __int_as_float(v >= 0 ? v : v ^ 0x7FFFFFFF);
  }
}
"""


def int_as_float_gpu(input):
    with torch.cuda.device_of(input):
        output = torch.cuda.FloatTensor(input.size())
        num_threads = input.numel()
        f = load_kernel('int_as_float', kernel, num_threads=num_threads)
        f(args=[input.data_ptr(), output.data_ptr()],
          block=(cuda_num_threads, 1, 1),
          grid=(get_blocks(num_threads), 1, 1),
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return output
