import torch

from ...sparse import SparseTensor
from .graph_conv import SparseMM


def cheb_conv(x, edge_index, weight, edge_attr=None, bias=None):
    row, col = edge_index
    n, e, K = x.size(0), row.size(0), weight.size(0)
    # raise NotImplementedError

    if edge_attr is None:
        edge_attr = x.data.new(e).fill_(1)

    # Compute degree.
    degree = x.data.new(n).fill_(0).scatter_add_(0, row, edge_attr)
    degree = degree.pow_(-0.5)

    # Compute normalized and rescaled Laplacian.
    edge_attr *= degree[row]
    edge_attr *= degree[col]
    lap = SparseTensor(edge_index, -edge_attr, torch.Size([n, n]))

    # Convolution.
    Tx_0 = x
    output = torch.mm(Tx_0, weight[0])

    if K > 1:
        Tx_1 = SparseMM(lap)(x)
        output += torch.mm(Tx_1, weight[1])

    for k in range(2, K):
        Tx_2 = 2 * SparseMM(lap)(Tx_1) - Tx_0
        output += torch.mm(Tx_2, weight[k])
        Tx_0, Tx_1 = Tx_1, Tx_2

    if bias is not None:
        output += bias

    return output
