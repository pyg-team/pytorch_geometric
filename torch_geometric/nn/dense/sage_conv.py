import torch


class DenseSAGEConv(torch.nn.Module):
    def __init__(self, aggregator, out_channels, bias=True):
        super(DenseSAGEConv, self).__init__()

    def forward(self, x, adj):
        pass

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.aggregator,
                                   self.weight.size(1))
