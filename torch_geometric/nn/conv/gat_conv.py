import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from ..inits import uniform


class GATConv(torch.nn.Module):
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
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att_weight = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_channels * heads))
        elif bias and not concat:
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
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.mm(x, self.weight)
        x = x.view(-1, self.heads, self.out_channels)

        # Add self-loops to adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        # Compute attention coefficients.
        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = (alpha * self.att_weight).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, row, num_nodes=x.size(0))

        # Sample attention coefficients stochastically.
        dropout = self.dropout if self.training else 0
        alpha = F.dropout(alpha, p=dropout, training=True)

        # Sum up neighborhoods.
        out = alpha.view(-1, self.heads, 1) * x[col]
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.sum(dim=1) / self.heads

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
