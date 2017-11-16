import torch


def identity(size):
    y, x = size

    value = torch.ones(min(y, x))
    index = torch.arange(0, min(y, x), out=torch.LongTensor())
    index = index.repeat(2, 1)

    return torch.sparse.FloatTensor(index, value, size)
