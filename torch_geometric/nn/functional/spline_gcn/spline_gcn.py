import torch
from torch.autograd import Variable

from .spline import spline

from .edgewise_spline_weighting import edgewise_spline_weighting


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
    amount, index = spline(values, kernel_size, is_open_spline, degree)
    output = edgewise_spline_weighting(output, weight[:-1], amount, index)
    print(output)

    # Convolution via `scatter_add`. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    zero = output.data.new(adj.size(1), output.size(1)).fill_(0.0)
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    row = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, Variable(row), output)

    # Weighten root node features by multiplying with the meaned weights from
    # the origin.
    output += torch.mm(input, weight[-1])

    if bias is not None:
        output += bias

    return output
