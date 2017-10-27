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
const ${Dtype}* input, const ${Dtype}* weight, const ${Dtype}* amount,
const long* index) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    const int e_idx = idx / ${M_out};
    const int m_out_idx = idx % ${M_out};

    ${Dtype} w;
    ${Dtype} g;
    ${Dtype} f;
    ${Dtype} w_grad;
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
      }
    }
  }
}
'''


class EdgewiseSplineWeightingGPU(Function):
    def __init__(self, amount, index):
        super(EdgewiseSplineWeightingGPU, self).__init__()
        assert amount.is_cuda and index.is_cuda
        self.amount = amount
        self.index = index

    def forward(self, input, weight):
        assert input.is_cuda and weight.is_cuda

        self.save_for_backward(input, weight)

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
        k_max = self.amount.size(1)
        num_edges = input.size(0)

        grad_input = grad_output.new(num_edges, M_in).fill_(0)
        grad_weight = grad_output.new(K, M_in, M_out).fill_(0)

        num_threads = grad_output.numel()

        with torch.cuda.device_of(grad_output):
            f = load_kernel(
                'edgewise_spline_weighting_backward_kernel',
                _edgewise_spline_weighting_backward_kernel,
                Dtype=Dtype(input),
                num_threads=num_threads,
                num_edges=num_edges,
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
                  input.data_ptr(),
                  weight.data_ptr(),
                  self.amount.data_ptr(),
                  self.index.data_ptr()
              ],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight
