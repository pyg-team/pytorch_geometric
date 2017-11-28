import torch
from torch.autograd import Variable

from .spline import spline

from .edgewise_spline_weighting_bp2adj_gpu import EdgewiseSplineWeightingGPU


def spline_conv_bp2adj(
        adj,  # Tensor
        input,  # Variable
        weight,  # Variable
        kernel_size,
        is_open_spline,
        K,
        degree=1,
        bias=None):

    values = adj._values()
    row, col = adj._indices()

    # Get features for every end vertex with shape [|E| x M_in].
    output = input[col]
    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    output = EdgewiseSplineWeightingGPU(kernel_size, is_open_spline, K)(output, weight[:-1], values)

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
