import torch
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, scatter_

from ..inits import uniform


class GraphConv(torch.nn.Module):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta} \mathbf{x}_j.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggr (string): The aggregation operator to use (one of :obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, aggr='add', bias=True):
        super(GraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.root = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index

        out = torch.mm(x, self.weight)
        out = scatter_(self.aggr, out[col], row, dim_size=x.size(0))
        out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
