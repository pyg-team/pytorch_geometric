import torch
from torch.autograd import Variable

from .new import new


def matmul(edge_index, edge_attr, tensor):
    tensor = tensor if tensor.dim() > 1 else tensor.unsqueeze(-1)
    assert edge_attr.dim() == 1 and tensor.dim() == 2

    num_nodes, dim = tensor.size()
    row, col = edge_index
    row = row if torch.is_tensor(tensor) else Variable(row)

    output_col = edge_attr.unsqueeze(-1) * tensor[col]
    output = new(output_col, num_nodes, dim).fill_(0)
    row_expand = row.unsqueeze(-1).expand_as(output_col)
    output.scatter_add_(0, row_expand, output_col)

    return output
