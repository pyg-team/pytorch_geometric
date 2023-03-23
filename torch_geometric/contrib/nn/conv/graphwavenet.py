from functools import reduce
from operator import mul

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv, GCNConv


class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, out_timesteps,
                 dilations=[1, 2, 1, 2, 1, 2, 1, 2], adptive_embeddings=10,
                 dropout=0.3, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512):

        super(GraphWaveNet, self).__init__()

        self.total_dilation = sum(dilations)
        self.num_dilations = len(dilations)
        self.num_nodes = num_nodes

        self.dropout = dropout

        self.e1 = nn.Parameter(torch.randn(num_nodes, adptive_embeddings),
                               requires_grad=True)
        self.e2 = nn.Parameter(torch.randn(adptive_embeddings, num_nodes),
                               requires_grad=True)

        self.input = nn.Conv2d(in_channels=in_channels,
                               out_channels=residual_channels,
                               kernel_size=(1, 1))

        self.tcn_a = nn.ModuleList()
        self.tcn_b = nn.ModuleList()
        self.gcn = nn.ModuleList()
        self.gcn_adp = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.skip = nn.ModuleList()

        self.out_timesteps = out_timesteps
        self.out_channels = out_channels

        for d in dilations:
            self.tcn_a.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=dilation_channels, kernel_size=(1, 2),
                          dilation=d))

            self.tcn_b.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=dilation_channels, kernel_size=(1, 2),
                          dilation=d))

            self.gcn.append(
                GCNConv(in_channels=dilation_channels,
                        out_channels=residual_channels))
            self.gcn_adp.append(
                DenseGCNConv(in_channels=dilation_channels,
                             out_channels=residual_channels))

            self.skip.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=skip_channels, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))

        self.end1 = nn.Conv2d(in_channels=skip_channels,
                              out_channels=end_channels, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(in_channels=end_channels,
                              out_channels=out_channels * out_timesteps,
                              kernel_size=(1, 1))

    def forward(self, x, edge_index, edge_weight=None):
        # x -> (*, t, num_nodes, in_channels)
        # edge_index -> (2, messages)
        # edge_weight -> (messages)

        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=0)

        batch_size = x.size()[:-3]
        input_timesteps = x.size(-3)

        x = x.transpose(-1, -3)

        if self.total_dilation + 1 > input_timesteps:
            x = F.pad(x, (self.total_dilation - input_timesteps + 1, 0))

        x = self.input(x)

        adp = F.softmax(F.relu(torch.mm(self.e1, self.e2)), dim=1)
        edge_index_adp = [[], []]
        edge_weight_adp = []

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adp[i][j] != 0:
                    edge_index_adp[0].append(i)
                    edge_index_adp[1].append(j)
                    edge_weight_adp.append(adp[i][j])

        edge_index_adp = torch.tensor(edge_index_adp).to(adp.device)
        edge_weight_adp = torch.tensor(edge_weight_adp).to(adp.device)

        skip_out = None

        for k in range(self.num_dilations):
            residual = x

            g1 = self.tcn_a[k](x)
            g1 = torch.tanh(g1)

            g2 = self.tcn_b[k](x)
            g2 = torch.sigmoid(g2)

            g = g1 * g2

            skip_cur = self.skip[k](g)

            if skip_out == None:
                skip_out = skip_cur
            else:
                skip_out = skip_out[..., -skip_cur.size(-1):] + skip_cur

            g = g.transpose(-1, -3)

            timesteps = g.size(-3)
            g = g.reshape(reduce(mul, g.size()[:-2]), *g.size()[-2:])

            data = self.batch_timesteps(g, edge_index,
                                        edge_weight).to(g.device)

            gcn_out = self.gcn[k](data.x, data.edge_index,
                                  edge_weight=torch.flatten(data.edge_attr))
            gcn_out = gcn_out.reshape(*batch_size, timesteps, -1,
                                      self.gcn[k].out_channels)

            gcn_out_adp = self.gcn_adp[k](g, adp)
            gcn_out_adp = gcn_out_adp.reshape(*batch_size, timesteps, -1,
                                              self.gcn[k].out_channels)

            x = gcn_out + gcn_out_adp

            x = F.dropout(x, p=self.dropout)

            x = x.transpose(-3, -1)

            x = x + residual[..., -x.size(-1):]
            x = self.bn[k](x)

        skip_out = skip_out[..., -1:]

        x = torch.relu(skip_out)
        x = self.end1(x)
        x = torch.relu(x)
        x = self.end2(x)
        x = x.reshape(*batch_size, self.out_timesteps, self.out_channels,
                      self.num_nodes).transpose(-1, -2)

        return x

    def batch_timesteps(self, x, edge_index, edge_weight=None):
        edge_index = edge_index.expand(x.size(0), *edge_index.shape)

        if edge_weight != None:
            edge_weight = edge_weight.expand(x.size(0), *edge_weight.shape)

        dataset = [
            Data(x=_x, edge_index=e_i, edge_attr=e_w)
            for _x, e_i, e_w in zip(x, edge_index, edge_weight)
        ]
        loader = DataLoader(dataset=dataset, batch_size=x.size(0))

        return next(iter(loader))
