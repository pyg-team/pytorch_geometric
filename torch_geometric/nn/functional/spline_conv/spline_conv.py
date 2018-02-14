import torch
from torch.autograd import Variable
import time

from .spline import spline

from .edgewise_spline_weighting import edgewise_spline_weighting


def spline_conv(
        adj,  # Tensor
        input,  # Variable
        weight,  # Variable
        kernel_size,
        is_open_spline,
        K,
        forward_kernel,
        backward_kernel,
        basis_kernel,
        degree=1,
        bias=None,):
    t_forward = time.process_time()
    if input.dim() == 1:
        input = input.unsqueeze(1)

    values = adj._values()
    row, col = adj._indices()

    # Get features for every end vertex with shape [|E| x M_in].
    t_gather = time.process_time()
    output = input[col]
    t_gather = time.process_time() - t_gather
    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].

    t_basis = time.process_time()
    amount, index = spline(values, kernel_size, is_open_spline, K, degree, basis_kernel)
    t_basis = time.process_time() - t_basis

    t_conv = time.process_time()
    output = edgewise_spline_weighting(output, weight[:-1], amount, index, forward_kernel, backward_kernel)
    t_conv = time.process_time() - t_conv
    print('t_gather',t_gather,'time_basis:',t_basis,'time_conv:',t_conv)

    # Convolution via `scatter_add`. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    t_scatter_add = time.process_time()
    zero = output.data.new(adj.size(1), output.size(1)).fill_(0.0)
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    r = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, Variable(r), output)
    t_scatter_add = time.process_time() - t_scatter_add

    # Weighten root node features by multiplying with root weight.

    t_root_weight = time.process_time()
    output += torch.mm(input, weight[-1])
    t_root_weight = time.process_time() - t_root_weight

    # Normalize output by degree.
    t_normalize = time.process_time()
    ones = values.new(values.size(0)).fill_(1)
    zero = values.new(output.size(0)).fill_(0)
    degree = zero.scatter_add_(0, row, ones)
    degree = torch.clamp(degree, min=1)
    output = output / Variable(degree.view(-1, 1))
    t_normalize = time.process_time() - t_normalize

    print('t_scatter_add:',t_scatter_add,'t_root_weight:',t_root_weight,'t_normalize:',t_normalize)

    if bias is not None:
        output += bias

    t_forward = time.process_time()- t_forward
    print('t_forward',t_forward)
    return output
