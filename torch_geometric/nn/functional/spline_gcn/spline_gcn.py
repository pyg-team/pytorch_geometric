from functools import reduce

import torch
from torch.autograd import Variable

from .spline import spline_weights
from .edgewise_spline_weighting_gpu import EdgewiseSplineWeighting


def spline_gcn(
        adj,  # Tensor
        input,  # Variable
        weight,  # Variable
        kernel_size,
        is_open_spline,
        degree=1,
        bias=None):

    values = adj._values()
    row, col = adj._indices()

    # Get features for every end vertex with shape [|E| x M_in].
    output = input[col]

    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    amount, index = spline_weights(values, kernel_size, is_open_spline, degree)
    op = EdgewiseSplineWeighting(amount, index)
    output = op(output, weight)
    # Convolution via `scatter_add`. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    zero = torch.zeros(adj.size(1), output.size(1)).type_as(output.data)
    zero = zero.cuda() if output.is_cuda else zero
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    row = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, Variable(row), output)

    # Weighten root node features by multiplying with the meaned weights from
    # the origin.
    root_index = torch.arange(0, reduce(lambda x, y: x * y, kernel_size[1:]))
    root_index = root_index.type_as(index)
    root_weight = weight[root_index].mean(0)
    output += torch.mm(input, root_weight)

    if bias is not None:
        output += bias

    return output
