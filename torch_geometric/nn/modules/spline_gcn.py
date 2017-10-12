import math

import torch
from torch.nn import Module, Parameter

# import torch_geometric.nn.functional as F


class SplineGCN(Module):
    """
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Mesh dimensions.
        kernel_size (int, tuple or triple): Size of the convolving kernel.
        spline_degree (int, tuple or triple): B-Spline degree.
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
            raise ValueError()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, )
        if len(kernel_size) < dim:
            # TODO: Fill with last value.
            pass

        if isinstance(spline_degree, int):
            spline_degree = (spline_degree, )
        if len(spline_degree) < dim:
            # TODO: Fill with last value.
            pass

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.spline_degree = spline_degree

        self.weight = Parameter(
            torch.Tensor(*kernel_size, in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
        return features

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', spline_degree={spline_degree}')
        if self.bias is None:
            s += 'bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
