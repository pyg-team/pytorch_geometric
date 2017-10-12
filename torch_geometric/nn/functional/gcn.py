import torch
from torch_geometric.sparse import mm


def gcn(adj, features, weight, bias):
    # TODO: add identy
    # TODO: Compute degree and normalized adj
    # TODO: Check if on cuda?

    output = mm(adj, features)
    output = torch.mm(output, weight)

    if bias is not None:
        output += bias
    return output
