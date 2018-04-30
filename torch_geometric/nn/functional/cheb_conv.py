import torch
from torch_geometric.utils import degree, matmul


def cheb_conv(x, edge_index, edge_attr, weight, bias=None):
    row, col = edge_index
    n, e, K = x.size(0), row.size(0), weight.size(0)

    edge_attr = x.new_full((e, ), 1) if edge_attr is None else edge_attr
    deg = degree(row, n, dtype=edge_attr.dtype, device=edge_attr.device)

    # Compute normalized and rescaled Laplacian.
    deg.pow_(-0.5)
    lap = -deg[row] * edge_attr * deg[col]

    # Convolution.
    Tx_0 = x
    out = torch.mm(Tx_0, weight[0])

    if K > 1:
        Tx_1 = matmul(edge_index, lap, x)
        out += torch.mm(Tx_1, weight[1])

    for k in range(2, K):
        Tx_2 = 2 * matmul(edge_index, lap, Tx_1) - Tx_0
        out += torch.mm(Tx_2, weight[k])
        Tx_0, Tx_1 = Tx_1, Tx_2

    if bias is not None:
        out += bias

    return out
