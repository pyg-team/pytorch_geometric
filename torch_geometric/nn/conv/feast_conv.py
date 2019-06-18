import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from ..inits import glorot, zeros


class FeaStConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=8,
                 bias=True,
                 **kwargs):
        super(FeaStConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.u = Parameter(torch.Tensor(in_channels, heads))
        self.c = Parameter(torch.Tensor(heads))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.u)
        zeros(self.c)
        zeros(self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j):
        # dim: x_i, [E, F_in];
        # with translation invariance
        q = torch.mm((x_i - x_j), self.u) + self.c
        q = F.softmax(q, dim=1)  # [E, heads]

        x_j = torch.mm(
            x_j, self.weight).view(-1, self.heads, self.out_channels)
        return x_j * q.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # sum heads
        aggr_out = aggr_out.sum(dim=1)

        if self.bias:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
