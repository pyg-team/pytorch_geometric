import math

import torch
from torch.nn import Parameter
import torch.nn.functional as F
import dgl.function as fn


class GCNConv(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels, bias=True):
        super(GCNConv, self).__init__()
        self.g = g
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = x * self.g.ndata['norm']
        self.g.ndata['x'] = x
        self.g.update_all(
            fn.copy_src(src='x', out='m'), fn.sum(msg='m', out='x'))
        x = self.g.ndata.pop('x')
        x = x * self.g.ndata['norm']
        if self.bias is not None:
            x = x + self.bias
        return x


class GCN(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(g, in_channels, 16)
        self.conv2 = GCNConv(g, 16, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)
