from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
import torch


class TAGConv(MessagePassing):
    r"""The topology adaptive graph convolutional networks operator from the
     `"Topology Adaptive Graph Convolutional Networks"
     <https://arxiv.org/abs/1710.10370>`_ paper
    .. math::
        \mathbf{X}^{\prime} = {\left(\sum_{k=0}^K \theta_{k} \mathbf{\hat{A}^k}
        \right)} \mathbf{X} ,

    where :math:`\mathbf{\hat{A}} = \mathbf{A}` denotes the adjacency matrix

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 K=1,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(TAGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.cached_result = None
        self.lin = Linear(in_channels*(self.K+1), out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        y = x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=torch.ones(
                edge_index.shape[1], device=edge_index.device))
            y = torch.cat((y, x), 1)
        out = self.lin(y)

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
