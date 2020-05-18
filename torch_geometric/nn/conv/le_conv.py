import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from torch_geometric.nn.inits import uniform


class LEConv(MessagePassing):
    r"""The Local Extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/pdf/1911.07979.pdf>`_ paper.
    It finds the importance of the central node with respect to the
    neighboring nodes using difference operator.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} \mathbf{A}_{i, j}
        (\mathbf{x}_i \mathbf{\Theta}_2 - \mathbf{x}_j \mathbf{\Theta}_3)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(LEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        h = torch.matmul(x, self.weight)
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, x.size(self.node_dim))

        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, x_i, h_j, edge_weight):
        if edge_weight is None:
            return (self.lin1(x_i) - h_j)
        else:
            return edge_weight.view(-1, 1) * (self.lin1(x_i) - h_j)

    def update(self, aggr_out, x):
        return aggr_out + self.lin2(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
