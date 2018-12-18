import torch
from torch.nn import Parameter
from torch_sparse import spmm
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, add_self_loops

from ..inits import glorot, zeros


class GCNConv(torch.nn.Module):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.size(1), ))
        assert edge_attr.dim() == 1 and edge_attr.numel() == edge_index.size(1)

        # Add self-loops to adjacency matrix.
        edge_index = add_self_loops(edge_index, x.size(0))
        loop_value = x.new_full((x.size(0), ), 1 if not self.improved else 2)
        edge_attr = torch.cat([edge_attr, loop_value], dim=0)

        # Normalize adjacency matrix.
        row, col = edge_index
        deg = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        edge_attr = deg[row] * edge_attr * deg[col]

        # Perform the convolution.
        out = torch.mm(x, self.weight)
        out = spmm(edge_index, edge_attr, out.size(0), out)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
