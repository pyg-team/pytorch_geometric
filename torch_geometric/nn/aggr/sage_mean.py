import torch
from torch.nn import Parameter

from ..inits import uniform


class SAGEMeanAggr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SAGEMeanAggr, self).__init__()

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        row, col = edge_index

        out = x

        return out

    def __repr__(self):
        return '{}{}{}({}, {})'.format(self.__class__.__name__,
                                       self.weight.size(0),
                                       self.weight.size(1))
