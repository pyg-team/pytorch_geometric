import torch
from torch_geometric.utils import scatter_

from ..inits import reset


class EdgeConv(torch.nn.Module):
    r"""The edge convolutional operator

    .. math::
        x^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\Theta}(x_i \, \Vert \, x_j - x_i)

    with a MLP :math:`h_{\Theta}` from the `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.

    Args:
        nn (nn.Sequential): Neural network.
        aggr (string): The aggregation to use (one of :obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"add"`)
    """

    def __init__(self, nn, aggr='add'):
        super(EdgeConv, self).__init__()
        self.nn = nn
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index):
        """"""
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        out = torch.cat([x[row], x[col] - x[row]], dim=1)
        out = self.nn(out)
        out = scatter_(self.aggr, out, row, dim_size=x.size(0))

        return out

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.nn)
