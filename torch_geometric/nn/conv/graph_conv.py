import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import degree

from ..inits import uniform


class GraphConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super(GraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        row, col = edge_index

        out = torch.mm(x, self.weight)
        out = scatter_add(out[col], row, dim=0, dim_size=x.size(0))

        # Normalize by node degree (if wished).
        if self.norm:
            deg = degree(row, x.size(0), x.dtype)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if self.root is not None:
            out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
