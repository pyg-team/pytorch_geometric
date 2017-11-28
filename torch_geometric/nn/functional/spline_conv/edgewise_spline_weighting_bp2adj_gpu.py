import torch
from torch.autograd import Function

from ....utils.cuda import (cuda_num_threads, Stream, Dtype, load_kernel,
                            kernel_loop, get_blocks)

_edgewise_spline_weighting_forward_kernel = kernel_loop + '''
extern "C"
__global__ void edgewise_spline_weighting_forward_kernel(
const ${Dtype}* input, const ${Dtype}* weight, ${Dtype}* output,
const ${Dtype}* amount, const long* index) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    const int e_idx = idx / ${M_out};
    const int m_out_idx = idx % ${M_out};

    ${Dtype} result = 0.0;
    ${Dtype} w;
    ${Dtype} f;
    int k;
    ${Dtype} b;
    long c;
    long w_idx;

    for (int k_idx = 0; k_idx < ${k_max}; k_idx++) {
      k = e_idx * ${k_max} + k_idx;
      b = amount[k];
      c = index[k];

      for (int m_in_idx = 0; m_in_idx < ${M_in}; m_in_idx++) {
        w_idx = c * ${M_out} * ${M_in} +
                m_in_idx * ${M_out} +
                m_out_idx;

        w = weight[w_idx];
        f = input[e_idx * ${M_in} + m_in_idx];

        result += b * w * f;
      }
    }

    output[idx] = result;
  }
}
'''

_edgewise_spline_weighting_backward_kernel = kernel_loop + '''
extern "C"
__global__ void edgewise_spline_weighting_backward_kernel(
const ${Dtype}* grad_output, ${Dtype}* grad_input, ${Dtype}* grad_weight,
${Dtype}* grad_b, const ${Dtype}* input, const ${Dtype}* weight,
const ${Dtype}* amount, const long* index) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    const int e_idx = idx / ${M_out};
    const int m_out_idx = idx % ${M_out};

    ${Dtype} w;
    ${Dtype} g;
    ${Dtype} f;
    ${Dtype} w_grad;
    ${Dtype} b_grad;
    int k;
    ${Dtype} b;
    long c;
    long w_idx;

    for (int k_idx = 0; k_idx < ${k_max}; k_idx++) {
      k = e_idx * ${k_max} + k_idx;
      b = amount[k];
      c = index[k];

      for (int m_in_idx = 0; m_in_idx < ${M_in}; m_in_idx++) {
        w_idx = c * ${M_out} * ${M_in} +
                m_in_idx * ${M_out} +
                m_out_idx;

        w = weight[w_idx];

        // Calculate input gradient.
        g = grad_output[e_idx * ${M_out} + m_out_idx];
        atomicAdd(&(grad_input[e_idx * ${M_in} + m_in_idx]), b * w * g);
        // This is inefficient: `reduce_sum` shouldn't be done like this.
        // Looping over `M_out` would be better to avoid the `atomicAdd`.

        // Calculate weight gradient.
        f = input[e_idx * ${M_in} + m_in_idx];
        w_grad = f * b * grad_output[e_idx * ${M_out} + m_out_idx];
        atomicAdd(&(grad_weight[w_idx]), w_grad);
        // Not so efficient either, but not avoidable.

        //Calculate gradient of B
        b_grad = f * w * g;
        atomicAdd(&(grad_b[k]), b_grad);
      }
    }
  }
}
'''


_spline_kernel = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, ${Dtype}* amount, long* index,
const long* kernel_size, const long* is_open_spline) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

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

      value = input[e_idx * ${dim} + d_idx] *
              (kernel_size[d_idx] - is_open_spline[d_idx]);

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

_spline_kernel_backwards = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, ${Dtype}* amount, long* index,
const long* kernel_size, const long* is_open_spline) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

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

      value = input[e_idx * ${dim} + d_idx] *
              (kernel_size[d_idx] - is_open_spline[d_idx]);

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


class EdgewiseSplineWeightingGPU(Function):
    def __init__(self,  kernel_size, is_open_spline, K):
        super(EdgewiseSplineWeightingGPU, self).__init__()
        assert kernel_size.is_cuda and is_open_spline.is_cuda
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline
        self.K = K

    def forward(self, input, values, weight):
        assert input.is_cuda and weight.is_cuda

        self.save_for_backward(input, weight)

        values = values.unsqueeze(1) if len(values.size()) < 2 else values
        num_edges, dim = values.size()
        k_max = 2 ** dim

        amount = values.new(num_edges, k_max)
        index = values.new(num_edges, k_max).long()
        num_threads = amount.numel()

        with torch.cuda.device_of(input):
            f = load_kernel(
                'spline_kernel',
                _spline_kernel,
                Dtype=Dtype(input),
                num_threads=num_threads,
                k_max=k_max,
                dim=dim,
                K=self.K)
            f(block=(cuda_num_threads, 1, 1),
              grid=(get_blocks(num_threads), 1, 1),
              args=[
                  values.data_ptr(),
                  amount.data_ptr(),
                  index.data_ptr(),
                  self.kernel_size.data_ptr(),
                  self.is_open_spline.data_ptr()
              ],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))


        self.amount = amount
        self.index = index

        _, M_in, M_out = weight.size()
        k_max = self.amount.size(1)

        output = input.new(input.size(0), M_out)
        num_threads = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel(
                'edgewise_spline_weighting_forward_kernel',
                _edgewise_spline_weighting_forward_kernel,
                Dtype=Dtype(input),
                num_threads=num_threads,
                M_in=M_in,
                M_out=M_out,
                k_max=k_max)
            f(block=(cuda_num_threads, 1, 1),
              grid=(get_blocks(num_threads), 1, 1),
              args=[
                  input.data_ptr(),
                  weight.data_ptr(),
                  output.data_ptr(),
                  self.amount.data_ptr(),
                  self.index.data_ptr()
              ],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return output

    def backward(self, grad_output):
        input, weight = self.saved_tensors

        K, M_in, M_out = weight.size()
        E, k_max = self.amount.size()

        grad_input = grad_output.new(input.size(0), M_in).fill_(0)
        grad_weight = grad_output.new(K, M_in, M_out).fill_(0)
        grad_b = grad_output.new(E, k_max).fill_(0)

        num_threads = grad_output.numel()

        with torch.cuda.device_of(grad_output):
            f = load_kernel(
                'edgewise_spline_weighting_backward_kernel',
                _edgewise_spline_weighting_backward_kernel,
                Dtype=Dtype(input),
                num_threads=num_threads,
                M_in=M_in,
                M_out=M_out,
                k_max=k_max,
                K=K)
            f(block=(cuda_num_threads, 1, 1),
              grid=(get_blocks(num_threads), 1, 1),
              args=[
                  grad_output.data_ptr(),
                  grad_input.data_ptr(),
                  grad_weight.data_ptr(),
                  grad_b.data_ptr(),
                  input.data_ptr(),
                  weight.data_ptr(),
                  self.amount.data_ptr(),
                  self.index.data_ptr()
              ],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))


        grad_b = grad_output.new(E, k_max).fill_(0)

        return grad_input, grad_weight, grad_u
