import torch
from torch.nn import Linear
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from typing import Optional


class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
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
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K` on first
            execution, and will use the cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    cached_num_edges: Optional[int]
    cached_result: Optional[torch.Tensor]
    has_cached_result: bool

    def __init__(self, in_channels, out_channels, K=1, cached=False, bias=True,
                 **kwargs):
        super(SGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached

        self.cached_num_edges = None
        self.cached_result = None
        self.has_cached_result = False

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None
        self.cached_num_edges = None
        self.has_cached_result = False

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
        if self.cached and self.has_cached_result:
            cached_num_edges = self.cached_num_edges
            assert cached_num_edges is not None
            if edge_index.size(1) != cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached:
            x = self.lin(x)

        if not self.cached or not self.has_cached_result:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                         edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x
            self.has_cached_result = True

        if self.cached:
            tempx = self.cached_result
            assert tempx is not None
            x = self.lin(tempx)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
