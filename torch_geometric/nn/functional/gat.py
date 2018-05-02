from __future__ import division

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, softmax


def gat(x, edge_index, concat, negative_slope, dropout, weight, att_weight,
        bias):
    num_nodes = x.size(0)
    m_in, k, m_out = weight.size()
    row, col = add_self_loops(edge_index, num_nodes)

    x = torch.mm(x, weight.view(m_in, k * m_out)).view(num_nodes, k, m_out)
    x_row, x_col = x[row], x[col]

    # Compute attention coefficients.
    alpha = torch.cat([x_row, x_col], dim=-1).unsqueeze(-2)
    alpha = torch.matmul(alpha, att_weight.unsqueeze(-1))
    alpha = alpha.squeeze(-1).squeeze(-1)
    alpha = F.leaky_relu(alpha, negative_slope)
    alpha = softmax(alpha, row, num_nodes)

    # Sample attention coefficients stochastically.
    alpha = F.dropout(alpha, p=dropout, training=True)

    # Apply attention.
    x_col = alpha.unsqueeze(-1) * x_col

    # Sum up neighborhoods.
    out = scatter_add(x_col, row, dim=0, dim_size=num_nodes)

    if concat is True:
        out = out.view(num_nodes, k * m_out)
    else:
        out = out.sum(dim=1) / k

    if bias is not None:
        out += bias

    return out
