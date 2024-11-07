import torch

from torch_geometric.nn import MessagePassing


class MixupConv(MessagePassing):
    def __init__(self, conv_layer, in_channels, out_channels, aggr='mean',
                 bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layer = conv_layer(in_channels, out_channels, bias=bias,
                                     **kwargs)

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.conv_layer.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x1, edge_index, x2):
        h1 = self.conv_layer(x1, edge_index)
        return h1 + self.lin(x2)

    def __repr__(self):
        return '{}(conv={}, in_channels={}, out_channels={})'.format(
            self.__class__.__name__, self.conv_layer.__class__.__name__,
            self.in_channels, self.out_channels)
