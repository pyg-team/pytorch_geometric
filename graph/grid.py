import torch
import numpy as np


def grid_adj(size, connectivity=4, tensor=torch.ByteTensor):
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

    rows = rows.view(-1, 1).repeat(1, connectivity)
    cols = rows + kernel
    rows = rows.view(-1)
    cols = cols.view(-1)

    i = torch.cat((rows, cols)).view(2, -1)
    print(i)

    # .repeat(connectivity)
    # print(cols)
    # print(rows)

    # print(kernel)


def grid_points(size, tensor=torch.LongTensor):
    """Return the regular grid points of a given shape with distance `1`."""

    h, w = size
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = tensor(x)
    y = tensor(y)
    points = tensor(h * w, 2)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        points = points.cuda()

    points[:, 0] = y.view(-1)
    points[:, 1] = x.view(-1)
    return points


size = torch.Size([1, 1])
a = grid_adj(size, 4)
print(a)
