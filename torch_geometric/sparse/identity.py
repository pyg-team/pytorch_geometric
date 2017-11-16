import torch


def identity(size):
    h, w = size
    n = min(h, w)

    value = torch.ones(n)
    index = torch.arange(0, n, out=torch.LongTensor())
    index = index.repeat(2, 1)

    return torch.sparse.FloatTensor(index, value, size)
