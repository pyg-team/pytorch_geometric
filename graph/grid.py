import torch
import numpy as np


def grid_adj_indices(size, connectivity=4):
    """Return the unweighted adjacency matrix of a regular grid."""

    h, w = size

    if connectivity == 4:
        kernel = [-w - 2, -1, 1, w + 2]
    elif connectivity == 8:
        kernel = [-w - 3, -w - 2, -w - 1, -1, 1, w + 1, w + 2, w + 3]
    else:
        raise ValueError()

    kernel = torch.LongTensor(kernel)

    n = (h + 1) * (w + 2) - 1
    rows = torch.arange(w + 3, n, out=torch.LongTensor())

    if torch.cuda.is_available():
        kernel = kernel.cuda()
        rows = rows.cuda()

    # Broadcast kernel and compute indices.
    rows = rows.view(-1, 1).repeat(1, connectivity)
    cols = rows + kernel
    rows = rows.view(-1)
    cols = cols.view(-1)
    indices = torch.cat((rows, cols)).view(2, -1)

    print(indices)

    # .repeat(connectivity)
    # print(cols)
    # print(rows)

    # print(kernel)


def grid_points(size):
    """Return the regular grid points of a given shape with distance `1`."""

    h, w = size
    x = torch.arange(0, w)
    y = torch.arange(0, h)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    x = x.repeat(h)
    y = y.view(-1, 1).repeat(1, w).view(-1)

    return torch.cat((y, x)).view(2, -1).t()


size = torch.Size([3, 2])
a = grid_points(size)
print(a)
