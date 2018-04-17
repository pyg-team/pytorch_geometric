import torch
from torch_geometric.utils import new


def one_hot(src, max_value=None, out=None):
    src_data = src if torch.is_tensor(src) else src.data
    max_value = src_data.max() + 1 if max_value is None else max_value
    sizes = src.size(0), max_value

    out = new(src.float()) if out is None else out
    out.resize_(*sizes) if torch.is_tensor(out) else out.data.resize_(*sizes)
    out.fill_(0)
    out[torch.arange(sizes[0], out=src_data.new()), src_data] = 1

    return out
