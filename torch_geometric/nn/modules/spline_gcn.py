from functools import reduce
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
    """
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Edge dimension.
        kernel_size (int or list): Size of the convolving kernel.
        is_open_spline (bool or list, optional): Whether to use open or closed
            spline curves. (default `True`)
        spline_degree (int, optional): Spline degree. (default: 1)
        bias (bool, optional): If set to False, the layer will not learn an
            additive bias. (default: `True`)
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
        kernel_size = _repeat_last_to_count(kernel_size, dim)
        self.kernel_size = torch.LongTensor(kernel_size)
        is_open_spline = _repeat_last_to_count(is_open_spline, dim)
        self.is_open_spline = torch.LongTensor(is_open_spline)
        self.degree = degree
        self.K = reduce(lambda x, y: x * y, kernel_size)

        weight = torch.Tensor(self.K, in_features, out_features)
        self.weight = Parameter(weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features * self.K)

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
        return spline_gcn(adj, features, self.weight, self.kernel_size,
                          self.max_radius, self.degree, self.bias)

    def __repr__(self):
        s = ('{name}({in_features}, {out_features}, kernel_size='
             '{kernel_size_list}, is_open_spline={is_open_spline_list}, '
             'degree={degree}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        self.kernel_size_list = list(self.kernel_size.numpy())
        self.is_open_spline_list = list(self.is_open_spline.numpy())
        return s.format(name=self.__class__.__name__, **self.__dict__)
