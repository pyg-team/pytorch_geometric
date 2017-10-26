from functools import reduce
import time
import torch
from torch.autograd import Variable

from .spline import spline_weights

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

    t = time.process_time()
    # Get features for every end vertex with shape [|E| x M_in].
    output = input[col]
    print('1',time.process_time()-t)
    t = time.process_time()
    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    amount, index = spline_weights(values, kernel_size, is_open_spline, degree)
    print('2',time.process_time()-t)
    t = time.process_time()
    output = edgewise_spline_weighting(output, weight, amount, index)
    print('3',time.process_time()-t)
    t = time.process_time()

    # Convolution via `scatter_add`. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    zero = output.data.new(adj.size(1), output.size(1)).fill_(0.0)
    zero = zero.cuda() if output.is_cuda else zero
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    row = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, Variable(row), output)

    print('4',time.process_time()-t)
    t = time.process_time()
    # Weighten root node features by multiplying with the meaned weights from
    # the origin.

    root_weight = weight[:kernel_size[1:].prod()].mean(0)
    output += torch.mm(input, root_weight)
    print('5',time.process_time()-t)
    t = time.process_time()

    if bias is not None:
        output += bias

    return output
