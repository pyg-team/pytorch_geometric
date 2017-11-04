import torch


def grid(size, connectivity=4, out=None):
    """Return the unweighted adjacency matrix of a regular grid."""
    h, w = size
    n = h * w

    # BOTTOM
    row = torch.arange(0, n - w).long()
    col = torch.arange(w, n).long()

    index = torch.stack([row, col], dim=0)
    weight = torch.ones(n - w)
    bottom = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))

    # RIGHT
    row = torch.arange(0, w - 1).long().repeat(h).view(h, -1)
    row += torch.arange(0, n, w).long().view(-1, 1)
    row = row.view(-1)
    col = row + 1

    index = torch.stack([row, col], dim=0)
    weight = torch.ones(n - h)
    right = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))

    a = bottom + right

    if connectivity == 8:
        # BOTTOM RIGHT
        row = torch.arange(0, w - 1).long().repeat(h - 1).view(h - 1, -1)
        row += torch.arange(0, n - w, w).long().view(-1, 1)
        row = row.view(-1)
        col = row + w + 1

        index = torch.stack([row, col], dim=0)
        weight = torch.ones(row.size(0)) + 1
        br = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))
        a += br

        # # TOP RIGHT
        row = torch.arange(0, w - 1).long().repeat(h - 1).view(h - 1, -1)
        row += w
        row += torch.arange(0, n - w, w).long().view(-1, 1)
        row = row.view(-1)
        col = row - w + 1
        index = torch.stack([row, col], dim=0)
        weight = torch.ones(row.size(0)) + 1
        tr = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))
        a += tr

    a = a + a.t()
    return a.transpose(0, 1).coalesce()


def grid_position(size):
    """Return the regular grid points of a given shape with distance `1`."""

    h, w = size
    x = torch.arange(0, w)
    y = torch.arange(0, h)

    x = x.repeat(h)
    y = y.view(-1, 1).repeat(1, w).view(-1)

    return torch.cat((x, y)).view(2, -1).t()
