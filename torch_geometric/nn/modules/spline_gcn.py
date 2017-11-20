import math

import torch
from torch.nn import Module, Parameter

from ..functional.spline_gcn import spline_gcn


def _repeat_last_to_count(input, dim):
    if not isinstance(input, list):
        input = [input]

    if len(input) > dim:
        raise ValueError()

    if len(input) < dim:
        rest = dim - len(input)
        fill_value = input[len(input) - 1]
        input += [fill_value for _ in range(rest)]

    return input


class SplineGCN(Module):
    """Spline-based Convolutional Operator.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Size of the convolving kernel.
        is_open_spline (bool or [bool], optional): Whether to use open or
            closed B-spline bases. (default ``True``)
        degree (int, optional): B-spline degree. (default: ``1``)
        bias (bool, optional): If set to ``False``, the layer will not learn an
            additive bias. (default: ``True``)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 bias=True):

        super(SplineGCN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size_repr = _repeat_last_to_count(kernel_size, dim)
        kernel_size = torch.LongTensor(self.kernel_size_repr)
        self.register_buffer('kernel_size', kernel_size)
        self.is_open_spline_repr = _repeat_last_to_count(is_open_spline, dim)
        is_open_spline = torch.LongTensor(self.is_open_spline_repr)
        self.register_buffer('is_open_spline', is_open_spline)
        self.degree = degree
        self.K = self.kernel_size.prod()

        weight = torch.Tensor(self.K + 1, in_features, out_features)
        self.weight = Parameter(weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features * (self.K + 1))

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        return spline_gcn(
            adj, input, self.weight, self._buffers['kernel_size'],
            self._buffers['is_open_spline'], self.K, self.degree, self.bias)

    def __repr__(self):
        s = ('{name}({in_features}, {out_features}, kernel_size='
             '{kernel_size_repr}, is_open_spline={is_open_spline_repr}, '
             'degree={degree}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
