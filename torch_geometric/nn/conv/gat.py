import torch
from torch.nn import Module, Parameter

from ..inits import uniform
from ..functional.gat import gat


class GAT(Module):
    """Graph Attentional Layer from the `"Graph Attention Networks (GAT)"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): Whether to concat or average multi-head
            attentions (default: :obj:`True`)
        negative_slope (float, optional): LeakyRELU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout propbability of the normalized
            attention coefficients, i.e. exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GAT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels, heads, out_channels))
        self.att_weight = Parameter(torch.Tensor(heads, 2 * out_channels))

        if bias:
            if concat:
                self.bias = Parameter(torch.Tensor(out_channels * heads))
            else:
                self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.heads * self.in_channels
        uniform(size, self.weight)
        uniform(size, self.att_weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        dropout = self.dropout if self.training else 0
        return gat(x, edge_index, self.concat, self.negative_slope, dropout,
                   self.weight, self.att_weight, self.bias)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
