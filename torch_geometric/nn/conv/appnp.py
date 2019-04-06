from torch_geometric.nn.conv import MessagePassing, GCNConv


class APPNP(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, K, alpha, bias=True):
        super(APPNP, self).__init__('add')
        self.K = K
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, norm = GCNConv.norm(
            edge_index, x.size(0), edge_weight, dtype=x.dtype)

        hidden = x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = x * (1 - self.alpha)
            x = x + self.alpha * hidden

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)
