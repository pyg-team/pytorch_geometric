import torch
from torch.nn import Parameter

from ..inits import uniform
from ..prop import GCNProp


class GCNConv(torch.nn.Module):
    """Graph Convolutional Operator :math:`F_{out} = \hat{D}^{-1/2} \hat{A}
    \hat{D}^{-1/2} F_{in} W` with :math:`\hat{A} = A + I` and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` from the `"Semi-Supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prop = GCNProp()
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        out = torch.mm(x, self.weight)
        out = self.prop(out, edge_index, edge_attr)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
