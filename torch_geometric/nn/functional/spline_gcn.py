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
    output = zero.scatter_add_(0, Variable(row), output)

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
        self.save_for_backward(features, weight)
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

    def backward(self, features_grad_out):
        # features_grad_out: [|E| x M_out]
        # features_in: [|E|] x M_in]
        # weight: [K x M_in x M_out]
        features_in, weight = self.saved_tensors
        K, M_in, M_out = weight.size()

        features_grad_in = torch.zeros(features_grad_out.size(0), M_in)
        weight_grad_in = torch.zeros(weight.size())

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


def edgewise_spline_gcn(values,
                        features,
                        weight,
                        kernel_size,
                        max_radius,
                        degree=1):
    op = _EdgewiseSplineGcn(values, kernel_size, max_radius, degree)
    return op(features, weight)
