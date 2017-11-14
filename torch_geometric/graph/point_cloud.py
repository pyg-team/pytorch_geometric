import torch


def nn_graph(point, k=6):
    n, d = point.size()

    distance = point.contiguous().view(-1).repeat(n).view(n, n, d)
    distance -= point.unsqueeze(1)
    distance *= distance
    distance = distance.sum(dim=2)

    _, col = distance.sort(dim=1)
    col = col[:, 1:k + 1]
    col = col.contiguous().view(-1)

    row = torch.arange(0, n, out=torch.LongTensor())
    row = row.repeat(k).view(k, n).t()
    row = row.contiguous().view(-1)

    return torch.stack([row, col], dim=0)
