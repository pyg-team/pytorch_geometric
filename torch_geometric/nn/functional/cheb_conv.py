import torch

from .graph_conv import SparseMM


def cheb_conv(x, index, weight, bias=None):
    K = weight.size(0)
    row, col = index

    # Create normalized laplacian.
    lap = None

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
