import torch
from torch_geometric.utils import new


def one_hot(src, max_value=None, out=None):
    out = new(src).float() if out is None else out
    src = src if torch.is_tensor(src) else src.data
    max_value = src.max() + 1 if max_value is None else max_value
    sizes = src.size(0), max_value

    out.resize_(*sizes).fill_(0)
    out[torch.arange(sizes[0], out=src.new()), src] = 1

    return out
