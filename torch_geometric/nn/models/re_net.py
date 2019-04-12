import torch


class RENet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_rels,
                 aggr='mean',
                 seq_len=10,
                 dropout=0):
        super(RENet, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rels = num_rels
        self.aggr = aggr
        self.seq_len = seq_len
        self.dropout = dropout
