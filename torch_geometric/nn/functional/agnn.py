import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, softmax


def agnn(x, edge_index, beta):
    num_nodes = x.size(0)
    row, col = add_self_loops(edge_index, num_nodes)

    x_row, x_col = x[row], x[col]

    # Create sparse propagation matrix.
    norm = torch.norm(x, p=2, dim=1)
    norm = norm[row] * norm[col]
    cos = torch.matmul(x_row.unsqueeze(-2), x[col].unsqueeze(-1)).squeeze()
    cos /= norm
    cos = softmax(cos * beta, row, num_nodes)

    # Apply attention.
    x_col = cos.unsqueeze(-1) * x_col

    # Sum up neighborhoods.
    out = scatter_add(x_col, row, dim=0, dim_size=num_nodes)

    return out
