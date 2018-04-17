import torch
from torch.autograd import Variable
from torch_geometric.utils import new


def degree(index, num_nodes=None, out=None):
    num_nodes = index.max() + 1 if num_nodes is None else num_nodes
    out = index.new().float() if out is None else out
    index = index if torch.is_tensor(out) else Variable(index)

    if torch.is_tensor(out):
        out.resize_(num_nodes)
    else:
        out.data.resize_(num_nodes)

    one = new(out, index.size(0)).fill_(1)
    return out.fill_(0).scatter_add_(0, index, one)
