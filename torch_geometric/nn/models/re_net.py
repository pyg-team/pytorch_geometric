import math

import torch
from torch.nn import Parameter


class RENet(torch.nn.Module):
    def __init__(self,
                 num_nodes,
                 num_rels,
                 hidden_channels,
                 aggr='mean',
                 seq_len=10,
                 dropout=0):
        super(RENet, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_rels = num_rels
        self.aggr = aggr
        self.seq_len = seq_len
        self.dropout = dropout

        self.obj = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.rel = Parameter(torch.Tensor(num_rels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.obj, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.rel, gain=math.sqrt(2.0))

    def precompute_history(self, dataset):
        for i, data in enumerate(dataset):
            print(data)
            pass

    def forward(self, edge_index, rel, history):
        row, col = edge_index

        e_s = self.obj.index_select(0, row)
        e_r = self.rel.index_select(0, rel)

        # TODO:
        pass
