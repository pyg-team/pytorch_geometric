import torch


def eye(n, m=None):
    m = n if m is None else m
    min_n = min(n, m)

    value = torch.ones(min_n)
    index = torch.arange(0, min_n, out=torch.LongTensor())
    index = index.repeat(2, 1)

    return torch.sparse.FloatTensor(index, value, torch.Size([n, m]))
