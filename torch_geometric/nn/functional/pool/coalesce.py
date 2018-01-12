import torch
from torch_unique import unique_by_key


def remove_self_loops(index):
    row, col = index
    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)


def coalesce(index):
    e = (index.max() + 1) * index[0] + index[1]
    perm = torch.arange(0, e.size(0), out=e.new())

    _, perm = unique_by_key(e, perm)
    index = index[:, perm]
    index = index.contiguous()

    return index
