import torch
import torch.nn.functional as F
from torch.nn import Parameter

from ..inits import uniform


class DenseSAGEConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(DenseSAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, adj):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        out = torch.matmul(adj, x)

        if self.norm:
            out = out / adj.sum(dim=-1, keepdim=True)

        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
