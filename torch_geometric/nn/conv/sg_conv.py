from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing, GCNConv


class SGConv(MessagePassing):
    r"""The simgple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K=1, cached=False,
                 bias=True):
        super(SGConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.cached_result = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
