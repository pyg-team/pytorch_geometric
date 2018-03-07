import torch
from torch.nn import Module, Parameter as Param

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
        K (int): Chebyshev filter size, i.e. number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_features, out_features, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.K = K + 1
        self.weight = Param(torch.Tensor(self.K, in_features, out_features))

        if bias:
            self.bias = Param(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.K * self.in_features
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        return cheb_conv(x, edge_index, self.weight, edge_attr, self.bias)

    def __repr__(self):
        return repr(self)
