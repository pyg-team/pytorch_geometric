import torch
from torch.nn import Parameter
from torch_sparse import spmm
from torch_geometric.utils import degree

from ..inits import uniform


class ChebConv(torch.nn.Module):
    """Chebyshev Spectral Graph Convolutional Operator from the `"Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering"
    <https://arxiv.org/abs/1606.09375>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, i.e. number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K + 1, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        num_nodes, num_edges, K = x.size(0), row.size(0), self.weight.size(0)

        if edge_attr is None:
            edge_attr = x.new_ones((num_edges, ))

        deg = degree(row, num_nodes, dtype=x.dtype, device=x.device)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_attr * deg[col]

        # Perform filter operation recurrently .
        Tx_0 = x
        out = torch.mm(Tx_0, self.weight[0])

        if K > 1:
            Tx_1 = spmm(edge_index, lap, num_nodes, x)
            out = out + torch.mm(Tx_1, self.weight[1])

        for k in range(2, K):
            Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
            out = out + torch.mm(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0) - 1)
