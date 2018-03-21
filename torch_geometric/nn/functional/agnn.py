import torch


def agnn(x, edge_index, beta):
    row, col = edge_index

    # Create sparse propagation matrix.
    norm = torch.norm(x, 2, 1)
    return x

    # Attention-guided propagation.
