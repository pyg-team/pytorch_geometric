import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, segregate_self_loops

from ..inits import uniform


class GraphConv(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}_2 \mathbf{x}_j.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 aggr='add',
                 cached=False,
                 bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()
        self.cached_result = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight, _, edge_weight_of_self_loop = segregate_self_loops(
            edge_index, edge_weight, fill_value=fill_value, num_nodes=num_nodes)

        return edge_index, edge_weight, edge_weight_of_self_loop


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        h = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm, edge_weight_of_self_loop = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, h=h, norm=norm, edge_weight_of_self_loop=edge_weight_of_self_loop)

    def message(self, h_j, norm):
        return norm.view(-1, 1) * h_j

    def update(self, aggr_out, x, edge_weight_of_self_loop):
        return aggr_out + self.lin(x) * edge_weight_of_self_loop.view(-1, 1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

