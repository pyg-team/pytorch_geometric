import torch
from torch_unique import unique_by_key


def remove_self_loops(index):
    mask = (index[0] != index[1]).view(1, -1).expand(2, -1)
    return index[mask].view(2, -1)


def coalesce(index):
    e = (index.max() + 1) * index[0] + index[1]
    perm = torch.arange(0, e.size(0), out=e.new())

    _, perm = unique_by_key(e, perm)
    return index[:, perm]
