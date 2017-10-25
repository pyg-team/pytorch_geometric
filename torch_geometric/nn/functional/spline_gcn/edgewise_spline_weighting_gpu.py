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


_edgewise_spline_weighting_forward_kernel = kernel_loop + '''
extern "C"
__global__ void edgewise_spline_weighting_forward_kernel(
const ${Dtype}* input, const ${Dtype}* weight, ${Dtype}* output,
${Dtype}* amount, long* index) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    ${Dtype} result = 0.0;
    ${Dtype} w;
    ${Dtype} f;
    int k;
    ${Dtype} b;
    int c;
    int w_idx;

    // Compute output indices with index layout [|E|, M_out].
    const int e_idx = idx / ${M_out};
    const int m_out_idx = idx % ${M_out};

    for (int k_idx = 0; k_idx < ${k_max}; k_idx++) {
      k = e_idx * ${k_max} + k_idx;
      b = amount[k];
      c = index[k];

      for (int m_in_idx = 0; m_in_idx < ${M_in}; m_in_idx++) {
        w_idx = c * ${M_out} * ${M_in};
        w_idx += m_in_idx * ${M_out} + m_out_idx;

        w = weight[w_idx];
        f = input[e_idx * ${M_in} + m_in_idx];

        result += b * w * f;
      }
    }

    output[idx] = result;
  }
}
'''


class EdgewiseSplineWeighting(Function):
    def __init__(self, amount, index):
        super(EdgewiseSplineWeighting, self).__init__()
        assert amount.is_cuda and index.is_cuda
        self.amount = amount
        self.index = index

    def forward(self, input, weight):
        assert input.is_cuda and weight.is_cuda

        self.save_for_backward(input, weight)

        _, M_in, M_out = weight.size()
        k_max = self.amount.size(1)

        output = input.new(input.size(0), weight.size(2))
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

    def backawrd(self, grad_output):
        raise NotImplementedError()


_bspline_basis_backward_kernel = kernel_loop + '''
extern "C"
__global__ void bspline_basis_backward_kernel(
const ${Dtype}* input_grads, ${Dtype}* output_feature_grads, ${Dtype}* output_weights_grads,
                    const ${Dtype}* weights, const ${Dtype}* features, const float* amounts, const int* indices) {
  CUDA_KERNEL_LOOP(index, ${num_threads}) {

    //Initialize output with zero
    if(index<{num_edges$}*${M_in})
        output_feature_grads[index] = 0.0;
    for(int i=index; i<${K}*${M_in}*${M_out};i+=blockDim.x * gridDim.x)
        output_weights_grads[i] = 0.0;
    __syncthreads();

    // compute gradients
    const int e_idx = index / ${M_out};
    const int m_out_idx = index % ${M_out};
    ${Dtype} w;
    ${Dtype} f;
    ${Dtype} g;

    for(int k_idx=0;k_idx<${k_max};k_idx++)
    {
        const float b = amounts[e_idx*${k_max} + k_idx];
        const int c = indices[e_idx*${k_max} + k_idx];
        for(int m_in_idx = 0; i<${M_in}; i++)
        {
            // Nice coalescing memory access!
            w = weights[c*${M_out}*${M_in}+ m_in_idx*${M_out} + m_out_idx];

            // Calculate input feature gradients
            g = input_grads[e*${M_out} + m_out_idx];
            atomicAdd(&(output_feature_grads[e*${M_in} + m_in_idx]), b*g*w );
            // This is inefficient.... reduce_sum shouldnt be done like this...
            // Looping over M_out would be better to avoid the atomicAdd at least here...

            // Calculate weight gradient
            f = features[e_idx*${M_in} + m_in_idx];
            w_grad = f*b*input_grads[e*${M_out} + m_out_idx];
            atomicAdd(&(output_weights_grads[c*${M_out}*${M_in}+ m_in_idx*${M_out} + m_out_idx]), w_grad);
            // Not so efficient either... but not avoidable
        }
    }
}
            // This is only really fast... not super fast. But fast forward/inference is more important anyway!
'''

# class _EdgewiseSplineGcn_gpu(Function):
#     def __init__(self, values, kernel_size, max_radius, degree=1):
#         self.dim = len(kernel_size)
#         self.m = degree + 1
#         self.k = self.m**self.dim
#         amount, index = spline_weights(values, kernel_size, max_radius, degree)
#         self.amount = amount
#         self.index = index

#     def forward(self, features, weight):
#         # Number of edges
#         e = features.size(0)
#         _, M_in, M_out = weight.size()

#         features_out = features.new(e, M_out)
#         n = features_out.numel()
#         with torch.cuda.device_of(features):
#             f = load_kernel(
#                 'bspline_basis_forward_kernel',
#                 _bspline_basis_forward_kernel,
#                 Dtype=Dtype(input),
#                 num_edges=e,
#                 num_threads=n,
#                 M_in=M_in,
#                 M_out=M_out,
#                 k_max=self.k)
#             f(block=(CUDA_NUM_THREADS, 1, 1),
#               grid=(GET_BLOCKS(n), 1, 1),
#               args=[
#                   features.data_ptr(),
#                   weight.data_ptr(),
#                   features_out.data_ptr(),
#                   self.amount.data_ptr(),
#                   self.index.data_ptr()
#               ],
#               stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

#         self.save_for_backward(features, weight)
#         return features_out

#     def backward(self, features_grad_out):
#         # features_grad_out: [|E| x M_out]
#         # features_in: [|E|] x M_in]
#         # weight: [K x M_in x M_out]
#         features_in, weight = self.saved_tensors
#         K, M_in, M_out = weight.size()
#         e = features_grad_out.size(0)

#         features_grad_in = features_grad_out.new(e, M_in)
#         weight_grad_in = features_grad_out.new(K, M_in, M_out)
#         n = features_grad_in.numel() * self.k
#         with torch.cuda.device_of(features_grad_out):
#             f = load_kernel(
#                 'bspline_basis_backward_kernel',
#                 _bspline_basis_backward_kernel,
#                 Dtype=Dtype(input),
#                 num_edges=e,
#                 num_threads=n,
#                 M_in=M_in,
#                 M_out=M_out,
#                 k_max=self.k,
#                 K=K)
#             f(block=(CUDA_NUM_THREADS, 1, 1),
#               grid=(GET_BLOCKS(n), 1, 1),
#               args=[
#                   features_grad_out.data_ptr(),
#                   features_grad_in.data_ptr(),
#                   weight_grad_in.data_ptr(),
#                   weight.data_ptr(),
#                   features_in.data_ptr(),
#                   self.amount.data_ptr(),
#                   self.index.data_ptr()
#               ],
#               stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

#         return features_grad_in, weight_grad_in
