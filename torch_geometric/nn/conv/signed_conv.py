import torch
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing


class SignedConv(MessagePassing):
    r"""The signed graph convolutional operator from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{neg})}
        \left[ \frac{1}{|\mathcal{N}^{-}(v)|} \sum_{w \in \mathcal{N}^{-}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

    if :obj:`first_aggr` is set to :obj:`True`, and

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{pos})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{neg})} ,
        \mathbf{x}_v^{(\textrm{pos})} \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{neg})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{pos})} ,
        \mathbf{x}_v^{(\textrm{neg})} \right]

    otherwise.
    In case :obj:`first_aggr` is :obj:`False`, the layer expects :obj:`x` to be
    a tensor where :obj:`x[:, :in_channels]` denotes the positive node features
    :math:`\mathbf{X}^{(\textrm{pos})}` and :obj:`x[:, in_channels:]` denotes
    the negative node features :math:`\mathbf{X}^{(\textrm{neg})}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        first_aggr (bool): Denotes which aggregation formula to use.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, first_aggr, bias=True):
        super(SignedConv, self).__init__('mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos = Linear(2 * in_channels, out_channels, bias=bias)
            self.lin_neg = Linear(2 * in_channels, out_channels, bias=bias)
        else:
            self.lin_pos = Linear(3 * in_channels, out_channels, bias=bias)
            self.lin_neg = Linear(3 * in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_neg.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        """"""
        if self.first_aggr:
            assert x.size(1) == self.in_channels

            x_pos = torch.cat([self.propagate(pos_edge_index, x=x), x], dim=1)
            x_neg = torch.cat([self.propagate(neg_edge_index, x=x), x], dim=1)

        else:
            assert x.size(1) == 2 * self.in_channels

            x_1, x_2 = x[:, :self.in_channels], x[:, self.in_channels:]

            x_pos = torch.cat([
                self.propagate(pos_edge_index, x=x_1),
                self.propagate(neg_edge_index, x=x_2),
                x_1,
            ],
                              dim=1)

            x_neg = torch.cat([
                self.propagate(pos_edge_index, x=x_2),
                self.propagate(neg_edge_index, x=x_1),
                x_2,
            ],
                              dim=1)

        return torch.cat([self.lin_pos(x_pos), self.lin_neg(x_neg)], dim=1)

    def __repr__(self):
        return '{}({}, {}, first_aggr={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.first_aggr)
