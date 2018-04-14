from __future__ import division

import torch
from torch.autograd import Variable


def agnn(x, edge_index, beta):
    n, e, m = x.size(0), edge_index.size(1), x.size(1)
    row, col = edge_index

    # Create sparse propagation matrix.
    norm = torch.norm(x, 2, 1)
    norm = norm[row] * norm[col]
    cos = torch.matmul(x[row].unsqueeze(-2), x[col].unsqueeze(-1)).squeeze()
    cos /= norm + 1e-7  # Prevent underflow.
    cos *= beta

    # Scatter softmax.
    cos = cos.exp_()
    cos_sum = Variable(cos.data.new(n)).fill_(0)
    cos /= cos_sum.scatter_add_(0, Variable(row), cos)[row]

    # Apply attention.
    x_col = cos.unsqueeze(-1) * x[col]

    # Sum up neighborhoods.
    var_row = Variable(row.view(e, 1).expand(e, m))
    output = Variable(x.data.new(n, m)).fill_(0)
    output.scatter_add_(0, var_row, x_col)

    return output
