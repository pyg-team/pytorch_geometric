import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform
from .utils.repr import repr
from ..functional.cheb_conv import cheb_conv


class ChebConv(Module):
    """Chebyshev Spectral Graph Convolutional Operator from the `"Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering"
    <https://arxiv.org/abs/1606.09375>`_ paper.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        kernel_size (int): Chebyshev filter size, i.e. number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_features, out_features, kernel_size, bias=True):
        super(ChebConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        weight = torch.Tensor(kernel_size, in_features, out_features)
        self.weight = Parameter(weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.kernel_size * self.in_features
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, index):
        return cheb_conv(x, index, self.weight, self.bias)

    def __repr__(self):
        return repr(self)
