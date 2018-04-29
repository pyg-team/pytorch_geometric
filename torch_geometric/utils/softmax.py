from itertools import repeat

import torch
from torch.autograd import Variable
from torch_geometric.utils import new


def softmax(index, src, num_nodes=None):
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes

    index = index if torch.is_tensor(src) else Variable(index)
    sizes = list(src.size())[1:]

    output = src.exp()
    output_sum = new(output, num_nodes, *sizes).fill_(0)
    index_expand = index.view(-1, *repeat(1, len(sizes))).expand_as(src)
    output /= output_sum.scatter_add_(0, index_expand, output)[index]

    return output
