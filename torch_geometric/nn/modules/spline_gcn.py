import math

import torch
from torch.nn import Module, Parameter

from .._functions import spline_gcn


class SplineGCN(Module):
    """
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Mesh dimensions.
        kernel_size (int, tuple or triple): Size of the convolving kernel.
        spline_degree (int): B-Spline degree.
        bias (bool, optional): If set to False, the layer will not learn an
            additive bias. Default: `True`.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 dim,
                 kernel_size,
                 spline_degree,
                 bias=True):

        super(SplineGCN, self).__init__()

        if not 2 <= dim <= 3:
            raise ValueError('Node dimension is restricted to 2d or 3d.')

        # Fix kernel size representation to same length as dimensions.
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, )
        if len(kernel_size) < dim:
            kernel_size += tuple(kernel_size[len(kernel_size) - 1]
                                 for _ in range(0, dim - len(kernel_size)))

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.spline_degree = spline_degree

        weight_size = kernel_size + (in_features, out_features)
        self.weight = Parameter(torch.Tensor(*weight_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_features
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
        return spline_gcn(adj, features, self.weight, self.bias,
                          self.kernel_size, self.spline_degree)

    def __repr__(self):
        s = ('{name}({in_features}, {out_features}, kernel_size={kernel_size}'
             ', spline_degree={spline_degree}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
