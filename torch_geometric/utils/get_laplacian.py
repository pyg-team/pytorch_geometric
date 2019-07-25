import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce

from .loop import add_self_loops
from .num_nodes import maybe_num_nodes


def get_laplacian(edge_index, edge_weight=None, dtype=None,
                  num_nodes=None, normalization=None):
    """ Compute the graph Laplacian given the adjacency matrix

    Args:
        edge_index (torch.Tensor): edge indices representing the non-zero
            entries in the adjacency matrix of the graph
        edge_weight (torch.Tensor): the values of the non-zero entries of
            the adjacency matrix of the graph
        dtype (torch.dtype): the dtype to use for the edge_weight tensor
            in the case `edge_weight=None`
        num_nodes (int): number of nodes in the graph
        normalization: the normalization scheme for the graph laplacian;
            there are three possibilities:
            1. `None`: no normalization, L = D - A
            2. 'sym': symmetric normalization, L = 1 - D^{-1/2}.A.D^{-1/2}
            3. 'rw': random-walk normalization, L = 1 - -D^{-1}.A
    """

    assert normalization in [None, 'sym', 'rw'], "invalid normalization"

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 dtype=dtype,
                                 device=edge_index.device)

    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)

    # Compute the degree matrix D
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization is None:
        # L = D - A
        degree_index = torch.arange(0, num_nodes,
                                    dtype=torch.long,
                                    device=edge_index.device)
        degree_index = degree_index.unsqueeze(0).repeat(2, 1)

        edge_index = torch.cat([edge_index, degree_index], dim=1)
        edge_weight = torch.cat([-edge_weight, deg], dim=0)
    else:
        if normalization == 'sym':
            # Compute -A_norm = -D^{-1/2}.A.D^{-1/2}
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            edge_weight = -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'rw':
            # Compute -A_norm = -D^{-1}.A
            deg_inv = deg.pow(-1.0)
            deg_inv[deg_inv == float('inf')] = 0

            edge_weight = -deg_inv[row] * edge_weight
        else:
            raise ValueError("Normalization parameter invalid ({})".format(
                str(normalization)))

        # Compute the Laplacian L_norm = 1 - A_norm
        edge_index, edge_weight = add_self_loops(edge_index=edge_index,
                                                 edge_weight=edge_weight,
                                                 fill_value=1,
                                                 num_nodes=num_nodes)

    return coalesce(edge_index, edge_weight, num_nodes, num_nodes)
