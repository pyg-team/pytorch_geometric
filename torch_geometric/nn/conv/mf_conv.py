import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch.nn import Linear, ModuleList


class MFConv(MessagePassing):
    r"""The graph neural network operator from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}^{(\deg(i))} \left( \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    which trains a distinct weight matrix for each possible vertex degree.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        max_degree (int, optional): The maximum node degree to consider when
            updating weights (default: :obj:`10`)
        root_weight (bool, optional): If set to :obj:`True`, the layer will
            transform central node features differently than neighboring node
            features. (default: obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, max_degree=10,
                 root_weight=False, bias=True, **kwargs):
        super(MFConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree
        self.root_weight = root_weight

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=bias)
            for _ in range(max_degree + 1)
        ])

        if root_weight:
            self.root_lins = ModuleList([
                Linear(in_channels, out_channels, bias=False)
                for _ in range(max_degree + 1)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        if self.root_weight:
            for lin in self.root_lins:
                lin.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)

        deg = degree(edge_index[1 if self.flow == 'source_to_target' else 0],
                     x.size(0), dtype=torch.long)
        deg.clamp_(max=self.max_degree)

        if not self.root_weight:
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        h = self.propagate(edge_index, x=x)

        out = x.new_empty(list(x.size())[:-1] + [self.out_channels])

        for i in deg.unique().tolist():
            idx = (deg == i).nonzero().view(-1)

            r = self.rel_lins[i](h.index_select(self.node_dim, idx))
            if self.root_weight:
                r = r + self.root_lins[i](x.index_select(self.node_dim, idx))

            out.index_copy_(self.node_dim, idx, r)

        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
