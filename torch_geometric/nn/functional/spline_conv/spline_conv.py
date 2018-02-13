import torch
from torch.autograd import Variable

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

    if input.dim() == 1:
        input = input.unsqueeze(1)

    values = adj._values()
    row, col = adj._indices()

    # Get features for every end vertex with shape [|E| x M_in].
    output = input[col]
    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    amount, index = spline(values, kernel_size, is_open_spline, K, degree, basis_kernel)
    output = edgewise_spline_weighting(output, weight[:-1], amount, index, forward_kernel, backward_kernel)

    # Convolution via `scatter_add`. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    zero = output.data.new(adj.size(1), output.size(1)).fill_(0.0)
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    r = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, Variable(r), output)

    # Weighten root node features by multiplying with root weight.
    output += torch.mm(input, weight[-1])

    # Normalize output by degree.
    ones = values.new(values.size(0)).fill_(1)
    zero = values.new(output.size(0)).fill_(0)
    degree = zero.scatter_add_(0, row, ones)
    degree = torch.clamp(degree, min=1)
    output = output / Variable(degree.view(-1, 1))

    if bias is not None:
        output += bias

    return output
