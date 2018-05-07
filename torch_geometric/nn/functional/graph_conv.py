import torch
from torch_geometric.utils import degree, add_self_loops, matmul


def graph_conv(x, edge_index, edge_attr, weight, bias=None):
    row, col = edge_index
    n, e = x.size(0), row.size(0)

    edge_attr = x.new_ones((e, )) if edge_attr is None else edge_attr
    deg = degree(row, n, dtype=edge_attr.dtype, device=edge_attr.device)

    # Normalize and append adjacency matrix by self loops.
    deg.pow_(-0.5)
    edge_attr = deg[row] * edge_attr * deg[col]

    # Append self-loops.
    edge_attr = torch.cat([edge_attr, deg * deg], dim=0)
    edge_index = add_self_loops(edge_index, n)

    # Perform the convolution.
    out = torch.mm(x, weight)
    out = matmul(edge_index, edge_attr, out)

    if bias is not None:
        out += bias

    return out
