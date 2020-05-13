import torch
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_add

from torch_geometric.nn.inits import uniform


class LEConv(torch.nn.Module):
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
        num_nodes = x.shape[0]
        h = torch.matmul(x, self.weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=x.dtype,
                                     device=edge_index.device)
        edge_index, edge_weight = remove_self_loops(edge_index=edge_index,
                                                    edge_attr=edge_weight)
        deg = scatter_add(edge_weight, edge_index[0], dim=0,
                          dim_size=num_nodes)

        h_j = edge_weight.view(-1, 1) * h[edge_index[1]]
        aggr_out = scatter_add(h_j, edge_index[0], dim=0, dim_size=num_nodes)
        out = (deg.view(-1, 1) * self.lin1(x) + aggr_out) + self.lin2(x)
        edge_index, edge_weight = add_self_loops(edge_index=edge_index,
                                                 edge_weight=edge_weight,
                                                 num_nodes=num_nodes)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
