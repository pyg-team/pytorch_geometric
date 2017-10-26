import torch
from torch.autograd import Function

from ....utils.cuda import Dtype, Stream, load_kernel

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                        \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_spline_weights_kernel = kernel_loop + '''
extern "C"
__global__ void spline_weights_kernel(
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



class SplineWeightsGPU(Function):
    def __init__(self, kernel_size, is_open_spline, degree):
        super(SplineWeightsGPU, self).__init__()
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline
        self.degree = degree

    def forward(self, values):
        assert values.is_cuda

        _, M_in, M_out = weight.size()
        k_max = self.amount.size(1)

        output = input.new(input.size(0), M_out)
        num_threads = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel(
                'spline_weights_kernel',
                _spline_weights_kernel,
                Dtype=Dtype(input),
                num_threads=num_threads,
                M_in=M_in,
                M_out=M_out,
                k_max=k_max)
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(num_threads), 1, 1),
              args=[
                  input.data_ptr(),
                  weight.data_ptr(),
                  output.data_ptr(),
                  self.amount.data_ptr(),
                  self.index.data_ptr()
              ],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return output
