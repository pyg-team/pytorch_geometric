import time
from functools import reduce

import torch
from torch.autograd import Variable, Function

from .spline_utils import spline_weights
from collections import namedtuple
import cupy
import torch
from string import Template


def spline_gcn(
        adj,  # Tensor
        features,  # Variable
        weight,  # Variable
        kernel_size,
        max_radius,
        degree=1,
        bias=None):

    values = adj._values()
    row, col = adj._indices()

    # Get features for every end vertex with shape [|E| x M_in].
    output = features[col]

    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    output = edgewise_spline_gcn(values, output, weight, kernel_size,
                                 max_radius, degree)

    # Convolution via `scatter_add`. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    zero = torch.zeros(adj.size(1), output.size(1)).type_as(output.data)
    zero = zero.cuda() if output.is_cuda else zero
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    row = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, Variable(row), output)

    # Weighten root node features by multiplying with the meaned weights from
    # the origin.
    index = torch.arange(0, reduce(lambda x, y: x * y, kernel_size[1:])).long()
    index = index.cuda() if torch.cuda.is_available() else index
    root_weight = weight[index].mean(0)
    output += torch.mm(features, root_weight)

    if bias is not None:
        output += bias

    return output


class _EdgewiseSplineGcn(Function):
    def __init__(self, values, kernel_size, max_radius, degree=1):
        self.dim = len(kernel_size)
        self.m = degree + 1
        amount, index = spline_weights(values, kernel_size, max_radius, degree)
        self.amount = amount
        self.index = index

    def forward(self, features, weight):
        self.save_for_backward(features, weight)
        K, M_in, M_out = weight.size()

        features_out = torch.zeros(features.size(0), M_out).type_as(features)

        for k in range(self.m**self.dim):
            b = self.amount[:, k]  # [|E|]
            c = self.index[:, k]  # [|E|]

            for i in range(M_in):
                w = weight[:, i]  # [K x M_out]
                w = w[c]  # [|E| x M_out]
                f = features[:, i]  # [|E|]

                # Need to transpose twice, so we can make use of broadcasting.
                features_out += (f * b * w.t()).t()  # [|E| x M_out]

        return features_out

    def backward(self, features_grad_out):
        # features_grad_out: [|E| x M_out]
        # features_in: [|E|] x M_in]
        # weight: [K x M_in x M_out]
        features_in, weight = self.saved_tensors
        K, M_in, M_out = weight.size()
        n = features_in.size(0)

        features_grad_in = torch.zeros(n, M_in).type_as(features_in)
        weight_grad_in = torch.zeros(weight.size()).type_as(weight)

        for k in range(self.m**self.dim):
            b = self.amount[:, k]  # [|E|]
            c = self.index[:, k]  # [|E|]
            c_expand = c.contiguous().view(-1, 1).expand(c.size(0), M_out)

            for i in range(M_in):
                w = weight[:, i]  # [K x M_out]
                w = w[c]  # [|E| x M_out]

                f = b * torch.sum(features_grad_out * w, dim=1)  # [|E|]
                features_grad_in[:, i] += f

                f = features_in[:, i]  # [|E|]
                w_grad = (f * b * features_grad_out.t()).t()  # [|E|, M_out]
                weight_grad_in[:, i, :].scatter_add_(0, c_expand, w_grad)

        return features_grad_in, weight_grad_in



# GPU Version

Stream = namedtuple('Stream', ['ptr'])
CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


_bspline_basis_forward_kernel = kernel_loop + '''
extern "C"
__global__ void bspline_basis_forward_kernel(
const ${Dtype}* input_features, const ${Dtype}* weights, ${Dtype}* output_features, float* amounts, int* indices) {
  CUDA_KERNEL_LOOP(index, ${num_threads}) {

    //Initialize with zero
    ${Dtype} result = 0.0;

    // compute output features
    // expecting index layout [num_edges, M_out, k_max]
    const int e_idx = index / ${M_out};
    const int m_out_idx = index % ${M_out};


    ${Dtype} w;
    ${Dtype} f;
    for(int k_idx=0;k_idx<${k_max};k_idx++)
    {
        const float b = amounts[e_idx*${k_max} + k_idx];
        const int c = index[e_idx*${k_max} + k_idx];
        for(int m_in_idx = 0; i<${M_in}; i++)
        {
            // Nice coalescing memory access!
            w = weights[c*${M_out}*${M_in}+ m_in_idx*${M_out} + m_out_idx];

            // Same here, [M_in x E] would be better for memory access
            f = input_features[e*${M_in} + m_in_idx];

            result+=f*b*w;
        }
    }
    output_features[index] = result;

  }
}
            // This should be really fast... like super fast... with coalescing memory access hyper fast...
'''

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
        output_weights_grads[index] = 0.0;
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
            f = features[e_idx*{M_in} + m_in_idx];
            w_grad = f*b*input_grads[e*${M_out} + m_out_idx];
            atomicAdd(&(output_weights_grads[c*${M_out}*${M_in}+ m_in_idx*${M_out} + m_out_idx]), w_grad);

        }
    }
}
            // This is only really fast... not super fast. But fast forward/inference is more important anyway!
'''


class _EdgewiseSplineGcn_gpu(Function):
    def __init__(self, values, kernel_size, max_radius, degree=1):
        self.dim = len(kernel_size)
        self.m = degree + 1
        self.k = self.m**self.dim
        amount, index = spline_weights(values, kernel_size, max_radius, degree)
        self.amount = amount
        self.index = index

    def forward(self, features, weight):
        # Number of edges
        e = features.size(0)
        _, M_in, M_out = weight.size()

        features_out = features.new(e,M_out)
        n = features_out.numel()
        with torch.cuda.device_of(input):
            f = load_kernel('bspline_basis_forward_kernel', _bspline_basis_forward_kernel, Dtype=Dtype(input),
                            num_edges=e,num_threads=n, M_in=M_in, M_out=M_out, k_max=self.k)
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[features.data_ptr(), weight.data_ptr(), features_out.data_ptr(),
                    self.amount.data_ptr(),self.index.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        self.save_for_backward(features, weight)
        return features_out

    def backward(self, features_grad_out):
        # features_grad_out: [|E| x M_out]
        # features_in: [|E|] x M_in]
        # weight: [K x M_in x M_out]
        features_in, weight = self.saved_tensors
        K, M_in, M_out = weight.size()

        e = features_grad_out.size(0)
        _, M_in, M_out = weight.size()

        features_grad_in = features_grad_out.new(e,M_in)
        weight_grad_in = features_grad_out.new(K, M_in, M_out)
        n = features_grad_in.numel()*self.k
        with torch.cuda.device_of(input):
            f = load_kernel('bspline_basis_backward_kernel', _bspline_basis_backward_kernel, Dtype=Dtype(input),
                            num_edges=e,num_threads=n, M_in=M_in, M_out=M_out, k_max=self.k, K=K)
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[features_grad_out.data_ptr(), features_grad_in.data_ptr(), weight_grad_in.data_ptr(),
                    weight.data_ptr(), features_in.data_ptr(), self.amount.data_ptr(),self.index.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        '''
        for k in range(self.m**self.dim):
            b = self.amount[:, k]  # [|E|]
            c = self.index[:, k]  # [|E|]
            c_expand = c.contiguous().view(-1, 1).expand(c.size(0), M_out)

            for i in range(M_in):
                w = weight[:, i]  # [K x M_out]
                w = w[c]  # [|E| x M_out]

                f = b * torch.sum(features_grad_out * w, dim=1)  # [|E|]
                features_grad_in[:, i] += f

                f = features_in[:, i]  # [|E|]
                w_grad = (f * b * features_grad_out.t()).t()  # [|E|, M_out]
                weight_grad_in[:, i, :].scatter_add_(0, c_expand, w_grad)
        '''
        return features_grad_in, weight_grad_in



def edgewise_spline_gcn(values,
                        features,
                        weight,
                        kernel_size,
                        max_radius,
                        degree=1):
    op = _EdgewiseSplineGcn(values, kernel_size, max_radius, degree)
    return op(features, weight)
