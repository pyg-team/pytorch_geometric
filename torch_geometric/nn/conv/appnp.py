import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from typing import Optional


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
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    K: int
    alpha: float

    def __init__(self, K, alpha, bias=True, **kwargs):
        super(APPNP, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha

    def norm(self, edge_index, num_nodes: int,
             edge_weight: Optional[torch.Tensor] = None,
             improved: bool = False,
             dtype: Optional[int] = None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index[0], edge_index[1]
        assert edge_weight is not None
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        mask = (deg_inv_sqrt == float('inf'))
        zeros = torch.zeros(deg_inv_sqrt.shape,
                            dtype=deg_inv_sqrt.dtype,
                            device=deg_inv_sqrt.device)
        deg_inv_sqrt = torch.where(mask, zeros, deg_inv_sqrt)

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index,
                edge_weight: Optional[torch.Tensor] = None):
        """"""
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                     edge_weight, dtype=x.dtype)

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
