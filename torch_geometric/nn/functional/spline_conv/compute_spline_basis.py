import torch

from ....utils.cuda import (cuda_num_threads, Stream, Dtype, load_kernel,
                            kernel_loop, get_blocks)

_spline_kernel_linear = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, ${Dtype}* amount, long* index,
const long* kernel_size, const long* is_open_spline, int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads) {

    const int e_idx = idx / ${k_max};
    int k_idx = idx % ${k_max};

    int K = ${K};
    int k_idx_mod;
    int bot;
    int top;
    ${Dtype} value;
    ${Dtype} frac;
    ${Dtype} a = 1.0;
    long i = 0;

    for (int d_idx = 0; d_idx < ${dim}; d_idx++) {

      K /= kernel_size[d_idx];

      k_idx_mod = k_idx % 2;
      k_idx >>= 1;

      value = input[e_idx * ${dim} + d_idx];
      value *= kernel_size[d_idx] - is_open_spline[d_idx];

      frac = value - floor(value);

      a *= (1 - k_idx_mod) * frac + k_idx_mod * (1 - frac);

      bot = int(floor(value));
      top = (bot + 1) % kernel_size[d_idx];
      bot %= kernel_size[d_idx];
      i += (k_idx_mod * bot + (1 - k_idx_mod) * top) * K;
    }

    amount[idx] = a;
    index[idx] = i;
  }
}
'''

_spline_kernel_quadratic = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, ${Dtype}* amount, long* index,
const long* kernel_size, const long* is_open_spline, int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads) {

    const int e_idx = idx / ${k_max};
    int k_idx = idx % ${k_max};

    int K = ${K};
    int k_idx_mod;
    int pos;
    ${Dtype} value;
    ${Dtype} frac;
    ${Dtype} a = 1.0;
    long i = 0;

    for (int d_idx = 0; d_idx < ${dim}; d_idx++) {

      K /= kernel_size[d_idx];

      k_idx_mod = k_idx % 3;
      k_idx /= 3;

      value = input[e_idx * ${dim} + d_idx] *
              (kernel_size[d_idx] - (2 * is_open_spline[d_idx]));

      frac = value - floor(value);

      if (k_idx_mod == 0) a *= 0.5 * (1- frac) * (1-frac);
      else if (k_idx_mod == 1) a *= -frac * frac + frac + 0.5;
      else a *= 0.5 * frac * frac;

      pos = int(floor(value)) + k_idx_mod;
      pos %= kernel_size[d_idx];

      i += pos * K;
    }
    amount[idx] = a;
    index[idx] = i;
  }
}
'''

_spline_kernel_cubic = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, ${Dtype}* amount, long* index,
const long* kernel_size, const long* is_open_spline, int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads}) {

    const int e_idx = idx / ${k_max};
    int k_idx = idx % ${k_max};

    int K = ${K};
    int k_idx_mod;
    int pos;
    ${Dtype} value;
    ${Dtype} frac;
    ${Dtype} a = 1.0;
    long i = 0;

    for (int d_idx = 0; d_idx < ${dim}; d_idx++) {

      K /= kernel_size[d_idx];

      k_idx_mod = k_idx % 4;
      k_idx /= 4;

      value = input[e_idx * ${dim} + d_idx] *
              (kernel_size[d_idx] - (3 * is_open_spline[d_idx]));

      frac = value - floor(value);

      if (k_idx_mod == 0) a *= (1 - frac) * (1 - frac) * (1 - frac) / 6.0;
      else if (k_idx_mod == 1)
        a *= (3 * frac * frac * frac - 6 * frac * frac + 4) / 6.0;
      else if (k_idx_mod == 2)
        a *= (-3 * frac * frac * frac + 3 * frac * frac + 3 * frac + 1) / 6.0;
      else a *= frac * frac * frac / 6.0;

      pos = int(floor(value)) + k_idx_mod;
      pos %= kernel_size[d_idx];

      i += pos * K;
    }
    amount[idx] = a;
    index[idx] = i;
  }
}
'''


def get_basis_kernel(k_max, K, dim, degree):
    if degree == 3:
        _spline_kernel = _spline_kernel_cubic
    elif degree == 2:
        _spline_kernel = _spline_kernel_quadratic
    else:
        _spline_kernel = _spline_kernel_linear

    cuda_tensor = torch.FloatTensor([1]).cuda()
    with torch.cuda.device_of(cuda_tensor):
        f = load_kernel(
            'spline_kernel',
            _spline_kernel,
            Dtype='float',
            k_max=k_max,
            dim=dim,
            K=K)
    return f


def compute_spline_basis(input, kernel_size, is_open_spline, K, basis_kernel):
    assert input.is_cuda and kernel_size.is_cuda and is_open_spline.is_cuda

    input = input.unsqueeze(1) if len(input.size()) < 2 else input
    num_edges, dim = input.size()
    k_max = 2**dim

    amount = input.new(num_edges, k_max)
    index = input.new(num_edges, k_max).long()
    num_threads = amount.numel()

    with torch.cuda.device_of(input):
        basis_kernel(
            block=(cuda_num_threads, 1, 1),
            grid=(get_blocks(num_threads), 1, 1),
            args=[
                input.data_ptr(),
                amount.data_ptr(),
                index.data_ptr(),
                kernel_size.data_ptr(),
                is_open_spline.data_ptr(), num_threads
            ],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return amount, index
