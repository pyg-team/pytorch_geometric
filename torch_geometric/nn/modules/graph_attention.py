import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform
from .utils.repr import repr
from ..functional.graph_attention import graph_attention


class GraphAttention(Module):
    """Graph Attentional Layer from the `"Graph Attention Networks (GAT)"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
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
                 in_features,
                 out_features,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GraphAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_features, heads, out_features))
        self.att_weight = Parameter(torch.Tensor(heads, 2 * out_features))

        if bias:
            if concat:
                self.bias = Parameter(torch.Tensor(out_features * heads))
            else:
                self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.heads * self.in_features
        uniform(size, self.weight)
        uniform(size, self.att_weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        dropout = self.dropout if self.training else 0
        return graph_attention(x, edge_index, self.concat, self.negative_slope,
                               dropout, self.weight, self.att_weight,
                               self.bias)

    def __repr__(self):
        return repr(self)
