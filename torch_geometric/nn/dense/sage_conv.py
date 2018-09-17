import torch
from torch.nn import Parameter

from ..inits import uniform


class DenseSAGEMeanConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True):
        super(DenseSAGEMeanConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, adj):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        # Add self loops.
        arange = torch.arange(adj.size(-1), dtype=torch.long, device=x.device)
        adj[:, arange, arange] = 1

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
