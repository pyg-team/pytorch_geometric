from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .graph_conv import add_self_loops


def graph_attention(x,
                    edge_index,
                    concat,
                    negative_slope,
                    dropout,
                    weight,
                    att_weight,
                    bias=None):

    if weight.dim() == 2:
        weight = weight.unsqueeze(1)
    if att_weight.dim() == 1:
        att_weight = att_weight.unsqueeze(0)

    n, e = x.size(0), edge_index.size(1) + x.size(0)
    m_in, k, m_out = weight.size()

    # Append adjacency indices by self loops.
    edge_index = add_self_loops(edge_index, n)
    row, col = edge_index

    x = torch.mm(x, weight.view(m_in, k * m_out)).view(n, k, m_out)
    x_row, x_col = x[row], x[col]
    alpha = torch.cat([x_row, x_col], dim=-1)
    alpha = torch.matmul(alpha.unsqueeze(-2), att_weight.unsqueeze(-1))
    alpha = alpha.view(e, k)
    alpha = F.leaky_relu(alpha, negative_slope)

    # Scatter softmax.
    alpha = alpha.exp_()
    var_row = Variable(row.unsqueeze(-1).expand(e, k))
    alpha_sum = alpha.new(n, k).fill_(0).scatter_add_(0, var_row, alpha)
    alpha /= alpha_sum.gather(0, var_row)

    # Sample attention coefficients stochastically.
    alpha = F.dropout(alpha, p=dropout, training=True)

    # Apply attention.
    x_col = alpha.unsqueeze(-1) * x_col

    # Sum up neighborhoods.
    var_row = Variable(row.view(e, 1, 1).expand(e, k, m_out))
    output = x.new(n, k, m_out).fill_(0).scatter_add_(0, var_row, x_col)

    if concat is True:
        output = output.view(n, k * m_out)
    else:
        output = output.sum(dim=1)
        output /= k

    if bias is not None:
        output += bias

    return output
