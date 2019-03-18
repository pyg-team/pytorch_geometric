import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import EdgeSoftmax


class GATConv(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels, heads, dropout, slope):
        super(GATConv, self).__init__()
        self.g = g
        self.out_channels = out_channels
        self.heads = heads
        self.fc = Linear(in_channels, heads * out_channels, bias=False)
        self.attn_drop = torch.nn.Dropout(dropout)
        self.attn_l = Parameter(torch.Tensor(heads, out_channels, 1))
        self.attn_r = Parameter(torch.Tensor(heads, out_channels, 1))
        self.leaky_relu = torch.nn.LeakyReLU(slope)
        self.softmax = EdgeSoftmax()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        torch.nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        torch.nn.init.xavier_normal_(self.attn_r.data, gain=1.414)

    def forward(self, x):
        ft = self.fc(x).reshape((x.shape[0], self.heads, -1))  # NxHxD'
        head_ft = ft.transpose(0, 1)  # HxNxD'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)  # NxHx1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)  # NxHx1
        self.g.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})
        self.g.apply_edges(self.edge_attention)
        self.edge_softmax()
        self.g.update_all(
            fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        ret = self.g.ndata['ft'] / self.g.ndata['z']  # NxHxD'
        return ret.view(-1, self.heads * self.out_channels)

    def edge_attention(self, edges):
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a': a}

    def edge_softmax(self):
        scores, normalizer = self.softmax(self.g.edata['a'], self.g)
        self.g.ndata['z'] = normalizer
        self.g.edata['a_drop'] = self.attn_drop(scores)


class GAT(torch.nn.Module):
    def __init__(self, g, in_channels, out_channels):
        super(GAT, self).__init__()
        self.g = g
        self.conv1 = GATConv(g, in_channels, 8, 8, 0.6, 0.2)
        self.conv2 = GATConv(g, 64, out_channels, 1, 0.6, 0.2)

    def forward(self, x):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)
