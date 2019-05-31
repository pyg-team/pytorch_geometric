import torch
from torch_geometric.utils import scatter_
from torch_cluster import knn_graph

from ..inits import reset


class EdgeConv(torch.nn.Module):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (nn.Sequential): Neural network :math:`h_{\mathbf{\Theta}}`.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
    """

    def __init__(self, nn, aggr='max'):
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

        x_row, x_col = x.index_select(0, row), x.index_select(0, col)
        out = torch.cat([x_row, x_col - x_row], dim=1)
        out = self.nn(out)
        out = scatter_(self.aggr, out, row, dim_size=x.size(0))

        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class DynamicEdgeConv(torch.nn.Module):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (nn.Sequential): Neural network :math:`h_{\mathbf{\Theta}}`.
        k (int): Number of nearest neighbors.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
    """

    def __init__(self, nn, k, aggr='max'):
        super(DynamicEdgeConv, self).__init__()
        self.nn = nn
        self.k = k
        assert aggr in ['add', 'mean', 'max']
        aggr = 'sum' if aggr == 'add' else aggr
        self.aggr = getattr(torch, self.aggr)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, batch=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        row, col = knn_graph(x, self.k, batch, loop=False)
        x_row, x_col = x.index_select(0, row), x.index_select(0, col)
        out = torch.cat([x_row, x_col - x_row], dim=1)
        out = self.nn(out)
        out = self.aggr(out.view(-1, self.k, out.size(-1)), dim=1)

        return out

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)
