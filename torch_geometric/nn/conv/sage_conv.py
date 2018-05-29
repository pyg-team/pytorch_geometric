import torch
from torch.nn import Parameter

from ..inits import uniform


class SAGEConv(torch.nn.Module):
    def __init__(self, aggregator, out_channels, bias=True):
        super(SAGEConv, self).__init__()

        self.aggregator = aggregator
        in_channels = sum(list(self.aggregator.weight.size()))
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
        self.aggregator.reset_parameters()

    def forward(self, x, edge_index):
        out = torch.cat([x, self.aggregator(x, edge_index)], dim=-1)
        out = torch.matmul(out, self.weight)
        out = out / torch.norm(out, p=2, dim=-1).clamp(min=1).view(-1, 1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.aggregator,
                                   self.weight.size(1))
