from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing


class LEConv(MessagePassing):
    r"""The local extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper, which finds
    the importance of nodes with respect to their neighbors using the
    difference operator:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} a_{j,i}
        (\mathbf{x}_i \cdot \mathbf{\Theta}_2 - \mathbf{x}_j \cdot
        \mathbf{\Theta}_3)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(LEConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Linear(in_channels, out_channels, bias=bias)
        self.lin2 = Linear(in_channels, out_channels, bias=False)
        self.lin3 = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        a = self.lin1(x)
        b = self.lin2(x)

        out = self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight)
        return out + self.lin3(x)

    def message(self, a_i, b_j, edge_weight):
        out = a_i - b_j
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
