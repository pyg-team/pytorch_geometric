from functools import reduce

import torch
from torch.autograd import Variable, Function

from .spline_utils import spline_weights


def spline_gcn(
        adj,  # Tensor
        features,  # Variable
        weight,  # Parameter
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
    zero = torch.zeros(adj.size(1), output.size(1))
    zero = zero.cuda() if output.is_cuda else zero
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    row = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, row, output)

    # Weighten root node features by multiplying with the meaned weights from
    # the origin.
    index = torch.arange(0, reduce(lambda x, y: x * y, kernel_size[1:])).long()
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
        K, M_in, M_out = weight.size()

        features_out = torch.zeros(features.size(0), M_out)

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

    def backward(self, grad_output):
        pass


def edgewise_spline_gcn(values,
                        features,
                        weight,
                        kernel_size,
                        max_radius,
                        degree=1):
    op = _EdgewiseSplineGcn(values, kernel_size, max_radius, degree)
    return op(features, weight)
