import torch

from ....utils.cuda import (cuda_num_threads, Stream, Dtype, load_kernel,
                            kernel_loop, get_blocks)

_spline_kernel = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, ${Dtype}* amount, long* index,
const long* kernel_size, const long* is_open_spline) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    const int e_idx = idx / ${k_max};
    int k_idx = idx % ${k_max};

    int kernel_size_prod = ${kernel_size_prod};
    int k_idx_mod;
    int bot;
    int top;
    ${Dtype} value;
    ${Dtype} frac;
    ${Dtype} a = 1.0;
    long i = 0;

    for (int d_idx = 0; d_idx < ${dim}; d_idx++) {

      kernel_size_prod /= kernel_size[d_idx];

      k_idx_mod = k_idx % 2;
      k_idx >>= 1;

      value = input[e_idx * ${dim} + d_idx] *
              (kernel_size[d_idx] - is_open_spline[d_idx]);

      frac = value - floor(value);
      a *= (1 - k_idx_mod) * frac + k_idx_mod * (1 - frac);

      bot = int(floor(value));
      // bot %= kernel_size
      top = (bot + 1) % kernel_size[d_idx]; // - (1 - is_open_spline[d_idx]));
      bot = bot; // % kernel_size[d_idx]; // -(1-is_open_spline[d_idx]));
      i += (k_idx_mod * bot + (1 - k_idx_mod) * top) * kernel_size_prod;
    }

    amount[idx] = a;
    index[idx] = i;
  }
}
'''


def spline_gpu(input, kernel_size, is_open_spline, degree):
    assert input.is_cuda and kernel_size.is_cuda and is_open_spline.is_cuda
    assert degree == 1

    num_edges, dim = input.size()
    kernel_size_prod = kernel_size.prod()
    k_max = (degree + 1)**dim

    amount = input.new(num_edges, k_max)
    index = torch.cuda.LongTensor(num_edges, k_max)
    num_threads = amount.numel()

    with torch.cuda.device_of(input):
        f = load_kernel(
            'spline_kernel',
            _spline_kernel,
            Dtype=Dtype(input),
            num_threads=num_threads,
            k_max=k_max,
            dim=dim,
            kernel_size_prod=kernel_size_prod)
        f(block=(cuda_num_threads, 1, 1),
          grid=(get_blocks(num_threads), 1, 1),
          args=[
              input.data_ptr(),
              amount.data_ptr(),
              index.data_ptr(),
              kernel_size.data_ptr(),
              is_open_spline.data_ptr()
          ],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return amount, index
