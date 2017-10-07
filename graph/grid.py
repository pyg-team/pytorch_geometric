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
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    points = torch.FloatTensor(h * w, 2)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        points = points.cuda()

    points[:, 0] = y.view(-1)
    points[:, 1] = x.view(-1)
    return points


size = torch.Size([2, 2])
a = grid_adj(size, 4)
print(a)
