import torch
from torch.nn import Module, Parameter
from torch_geometric.utils import (degree, remove_self_loops, add_self_loops,
                                   matmul)

from .utils.inits import uniform


class GCNConv(Module):
    """Graph Convolutional Operator :math:`F_{out} = \hat{D}^{-1/2} \hat{A}
    \hat{D}^{-1/2} F_{in} W` with :math:`\hat{A} = A + I` and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` from the `"Semi-Supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        num_nodes, num_edges = x.size(0), row.size(0)

        if edge_attr is None:
            edge_attr = x.new_ones((num_edges, ))

        deg = degree(row, num_nodes, dtype=x.dtype, device=x.device)

        # Normalize adjacency matrix.
        deg = deg.pow(-0.5)
        edge_attr = deg[row] * edge_attr * deg[col]

        # Add self-loops to adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_attr = torch.cat([edge_attr, deg * deg], dim=0)
        edge_index = add_self_loops(edge_index, num_nodes)

        # Perform the convolution.
        out = torch.mm(x, self.weight)
        out = matmul(edge_index, edge_attr, out)

        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
