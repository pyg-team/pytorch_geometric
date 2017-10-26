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


class SplineWeightsGPU(Function):
    def __init__(self, kernel_size, is_open_spline, degree):
        super(SplineWeightsGPU, self).__init__()
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline
        self.degree = degree
        self.k_prod = kernel_size.prod()
        self._spline_weights_kernel = kernel_loop + '''
        extern "C"
        __global__ void spline_weights_kernel(
        const ${Dtype}* values, ${Dtype}* amounts, int* indices, int* kernel_size, int* is_open_spline) {
          CUDA_KERNEL_LOOP(idx, ${num_threads}) {
            int k = idx % k_max;
            int e_idx = idx / k_max;
            float prod = 1.0;
            int index;
            int k_prod = ${k_prod};
            for(int d_idx=0; d_idx <${d}; d_idx++)
            {
              k_prod = k_prod/kernel_size[d_idx];
              int value = values[e_idx * ${d} + d_idx] * (kernel_size[d_idx]-is_open_spline[d_idx]);
              int fraction = frac(value);
              prod *= (1-k%2)*fraction + k%2*(1-fraction);
              int bot = floor(value) % (kernel_size[d_idx]-(1-is_open_spline[d_idx]));
              int top = (bot+1) % (kernel_size[d_idx]-(1-is_open_spline[d_idx]));
              index += ((k%2)*bot + (1-k%2)*top)*k_prod;
              k = k >> 1;
            }
            amount[idx] = prod;
            indices[idx] = index;
          }
        }
        '''


    def forward(self, values):
        assert values.is_cuda

        k_max = self.amount.size(1)
        num_edges, d = values.size()

        amounts = values.new(num_edges, (self.degree+1)**d)
        indices = torch.cuda.IntTensor([num_edges, (self.degree+1)**d])
        num_threads = amounts.numel()

        with torch.cuda.device_of(input):
            f = load_kernel(
                'spline_weights_kernel',
                self._spline_weights_kernel,
                Dtype=Dtype(input),
                num_threads=num_threads,
                num_edges=num_edges,
                k_max=k_max,
                degree=self.degree+1,
                d=len(self.kernel_size.size()),
                k_prod = self.k_prod
            )
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(num_threads), 1, 1),
              args=[
                  values.data_ptr(),
                  amounts.data_ptr(),
                  indices.data_ptr(),
                  self.kernel_size.data_ptr(),
                  self.is_open_spline.data_ptr()
              ],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return amounts, indices
