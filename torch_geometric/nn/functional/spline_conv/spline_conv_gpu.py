import torch
from torch.autograd import Function

from ....utils.cuda import (cuda_num_threads, Stream, load_kernel, kernel_loop,
                            get_blocks)

_edgewise_spline_weighting_forward_kernel = kernel_loop + '''
extern "C"
__global__ void edgewise_spline_weighting_forward_kernel(
const ${Dtype}* input, const ${Dtype}* weight, ${Dtype}* output,
const ${Dtype}* amount, const long* index, int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads) {

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
const long* index, int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads) {

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
        w_grad = f * b * g;
        atomicAdd(&(grad_weight[w_idx]), w_grad);
        // Not so efficient either, but not avoidable.
      }
    }
  }
}
'''

_edgewise_spline_weighting_backward_kernel_bp2adj = kernel_loop + '''
extern "C"
__global__ void edgewise_spline_weighting_backward_kernel(
const ${Dtype}* grad_output, ${Dtype}* grad_input, ${Dtype}* grad_weight, 
${Dtype}* grad_amount, const ${Dtype}* input, const ${Dtype}* weight, 
const ${Dtype}* amount, const long* index, int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads) {

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
        w_grad = f * b * g;
        atomicAdd(&(grad_weight[w_idx]), w_grad);
        // Not so efficient either, but not avoidable.

        // Calculate B-spline basis tensor product gradient
        adj_g += g * f * w;
      }
      atomicAdd(&(grad_amount[e_idx,k_idx]), adj_g);
    }
  }
}
'''


def get_weighting_forward_kernel(M_in, M_out, k_max, bt_to_adj=False):
    cuda_tensor = torch.FloatTensor([1]).cuda()
    if bt_to_adj:
        kernel = _edgewise_spline_weighting_forward_kernel
    else:

        kernel = _edgewise_spline_weighting_forward_kernel
    with torch.cuda.device_of(cuda_tensor):
        f_fw = load_kernel(
            'edgewise_spline_weighting_forward_kernel',
            kernel,
            Dtype='float',
            M_in=M_in,
            M_out=M_out,
            k_max=k_max)
    return f_fw


def get_weighting_backward_kernel(M_in, M_out, k_max, K, bt_to_adj=False):
    cuda_tensor = torch.FloatTensor([1]).cuda()
    if bt_to_adj:
        kernel = _edgewise_spline_weighting_backward_kernel_bp2adj
    else:

        kernel = _edgewise_spline_weighting_backward_kernel
    with torch.cuda.device_of(cuda_tensor):
        f_bw = load_kernel(
            'edgewise_spline_weighting_backward_kernel',
            kernel,
            Dtype='float',
            M_in=M_in,
            M_out=M_out,
            k_max=k_max,
            K=K)
    return f_bw


_spline_kernel_linear = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, ${Dtype}* amount, long* index,
const long* kernel_size, const long* is_open_spline, int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads) {

    const int e_idx = idx / ${k_max};
    int k_idx = idx % ${k_max};

    int K = 1.0;
    int k_idx_mod;
    int bot;
    int top;
    ${Dtype} value;
    ${Dtype} frac;
    ${Dtype} a = 1.0;
    long i = 0;

    for (int d_idx = 0; d_idx < ${dim}; d_idx++) {
      k_idx_mod = k_idx % 2;
      k_idx >>= 1;

      value = input[e_idx * ${dim} + d_idx];
      if(value==1.0)
        value -= 0.000001;
      value *= kernel_size[d_idx] - is_open_spline[d_idx];
        
      frac = value - floor(value);

      a *= (1 - k_idx_mod) * (1 - frac) + k_idx_mod * frac;

      bot = int(floor(value));
      top = (bot + 1) % kernel_size[d_idx];
      bot %= kernel_size[d_idx];
      i += ((1 - k_idx_mod) * bot + k_idx_mod * top) * K;
      K *= kernel_size[d_idx];
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

_spline_kernel_linear_backward = kernel_loop + '''
extern "C"
__global__ void spline_kernel(
const ${Dtype}* input, const ${Dtype}* grad_amount, ${Dtype}* amount, 
${Dtype}* grad_adj, const long* kernel_size, const long* is_open_spline, 
int num_threads) {

  CUDA_KERNEL_LOOP(idx, num_threads) {

    const int e_idx = idx / ${dim};
    int d_idx = idx % ${dim};

    int K = ${K};
    int k_idx_mod;
    int bot;
    int top;
    ${Dtype} value;
    ${Dtype} frac;
    ${Dtype} grad_out = 0.0;
    long i = 0;
    
    int quotient = (int)pow(2.0,(float)d_idx)
    
    for (int k_idx = 0; k_idx < ${k_max}; k_idx++) {
      
      k_idx_mod = (k_idx/quotient) % 2;

      value = input[e_idx * ${dim} + d_idx];
      value *= kernel_size[d_idx] - is_open_spline[d_idx];

      frac = value - floor(value);
      residual = (1 - k_idx_mod) * frac + k_idx_mod * (1 - frac);
      int a_idx = e_idx*${k_max} + k_idx
      grad_out += grad_amount[a_idx]*amount[a_idx]/residual;

    }
    grad_adj[e_idx* ${dim} + d_idx] = grad_out;
  }
}
'''


def get_basis_kernel(k_max, K, dim, degree, bt_to_adj=False):
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


def get_basis_backward_kernel(k_max, K, dim, degree):
    if degree == 3:
        _spline_kernel = _spline_kernel_cubic
    elif degree == 2:
        _spline_kernel = _spline_kernel_quadratic
    else:
        _spline_kernel = _spline_kernel_linear_backward

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


class SplineConvGPU(Function):
    def __init__(self, kernel_size, is_open_spline, K, degree,
                               basis_kernel, basis_backward_kernel,
                               weighting_kernel, weighting_backward_kernel,
                               bp_to_adj=False):
        super(SplineConvGPU, self).__init__()
        self.degree = degree
        self.f_weighting_fw = weighting_kernel
        self.f_weighting_bw = weighting_backward_kernel
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline
        self.f_basis_fw = basis_kernel
        self.f_basis_bw = basis_backward_kernel
        self.bp_to_adj = bp_to_adj

    def forward(self, input, weight, adj_values):
        assert input.is_cuda and weight.is_cuda

        self.K, self.M_in, self.M_out = weight.size()

        # Compute B-spline basis tensor products
        adj_values = adj_values.unsqueeze(1) if len(adj_values.size()) < 2 \
            else adj_values
        num_edges, dim = adj_values.size()
        k_max = 2 ** dim

        amount = adj_values.new(num_edges, k_max)
        index = adj_values.new(num_edges, k_max).long()
        num_threads = amount.numel()

        with torch.cuda.device_of(input):
            self.f_basis_fw(
                block=(cuda_num_threads, 1, 1),
                grid=(get_blocks(num_threads), 1, 1),
                args=[
                    adj_values.data_ptr(),
                    amount.data_ptr(),
                    index.data_ptr(),
                    self.kernel_size.data_ptr(),
                    self.is_open_spline.data_ptr(), num_threads
                ],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        # Weight features
        output = input.new(input.size(0), self.M_out)
        num_threads = output.numel()

        with torch.cuda.device_of(input):
            self.f_weighting_fw(
                block=(cuda_num_threads, 1, 1),
                grid=(get_blocks(num_threads), 1, 1),
                args=[
                    input.data_ptr(),
                    weight.data_ptr(),
                    output.data_ptr(),
                    amount.data_ptr(),
                    index.data_ptr(), num_threads
                ],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        if self.bp_to_adj:
            self.save_for_backward(input, weight, adj_values, amount, index)
        else:
            self.save_for_backward(input, weight, amount, index)

        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(grad_output.size(0), self.M_in).fill_(0)
        grad_weight = grad_output.new(self.K, self.M_in, self.M_out).fill_(0)
        num_threads = grad_output.numel()

        if self.bp_to_adj:
            input, weight, adj_values, amount, index = self.saved_tensors
            grad_amount = grad_output.new(amount.size(0),
                                          amount.size(1)).fill_(0)
            with torch.cuda.device_of(grad_output):
                self.f_weighting_bw(
                    block=(cuda_num_threads, 1, 1),
                    grid=(get_blocks(num_threads), 1, 1),
                    args=[
                        grad_output.data_ptr(),
                        grad_input.data_ptr(),
                        grad_weight.data_ptr(),
                        grad_amount.data_ptr(),
                        input.data_ptr(),
                        weight.data_ptr(),
                        amount.data_ptr(),
                        index.data_ptr(), num_threads
                    ],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            num_threads = grad_amount.numel()
            grad_adj = grad_amount.new(grad_amount.size(0),
                                       self.kernel_size.size(0)).fill_(0)

            with torch.cuda.device_of(grad_amount):
                self.f_basis_bw(
                    block=(cuda_num_threads, 1, 1),
                    grid=(get_blocks(num_threads), 1, 1),
                    args=[
                        adj_values.data_ptr(),
                        grad_amount.data_ptr(),
                        amount.data_ptr(),
                        grad_adj.data_ptr(),
                        self.kernel_size.data_ptr(),
                        self.is_open_spline.data_ptr(), num_threads
                    ],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            return grad_input, grad_weight, grad_adj

        else:
            input, weight, amount, index = self.saved_tensors
            with torch.cuda.device_of(grad_output):
                self.f_weighting_bw(
                    block=(cuda_num_threads, 1, 1),
                    grid=(get_blocks(num_threads), 1, 1),
                    args=[
                        grad_output.data_ptr(),
                        grad_input.data_ptr(),
                        grad_weight.data_ptr(),
                        input.data_ptr(),
                        weight.data_ptr(),
                        amount.data_ptr(),
                        index.data_ptr(), num_threads
                    ],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            return grad_input, grad_weight
