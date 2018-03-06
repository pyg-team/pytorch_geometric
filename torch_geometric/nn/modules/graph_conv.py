import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform
from .utils.repr import repr
from ..functional.graph_conv import graph_conv


class GraphConv(Module):
    """Graph Convolutional Operator :math:`F_{out} = \hat{D}^{-1/2} \hat{A}
    \hat{D}^{-1/2} F_{in} W` with :math:`\hat{A} = A + I` and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` from the `"Semi-Supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_features
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, index):
        return graph_conv(x, index, self.weight, self.bias)

    def __repr__(self):
        return repr(self)
