import dgl.function as fn
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.inits import glorot, zeros


class GCNConv(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels):
        super().__init__()
        self.g = g
        self.weight = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def gcn_msg(self, edge):
        return {'m': edge.src['x'] * edge.src['norm']}

    def gcn_reduce(self, node):
        return {'x': node.mailbox['m'].sum(dim=1) * node.data['norm']}

    def forward(self, x):
        self.g.ndata['x'] = torch.matmul(x, self.weight)
        self.g.update_all(self.gcn_msg, self.gcn_reduce)
        x = self.g.ndata.pop('x')
        x = x + self.bias
        return x


class GCN(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(g, in_channels, 16)
        self.conv2 = GCNConv(g, 16, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)


class GCNSPMVConv(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels):
        super().__init__()
        self.g = g
        self.weight = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        self.g.ndata['x'] = x * self.g.ndata['norm']
        self.g.update_all(fn.copy_src(src='x', out='m'),
                          fn.sum(msg='m', out='x'))
        x = self.g.ndata.pop('x') * self.g.ndata['norm']
        x = x + self.bias
        return x


class GCNSPMV(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNSPMVConv(g, in_channels, 16)
        self.conv2 = GCNSPMVConv(g, 16, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)
