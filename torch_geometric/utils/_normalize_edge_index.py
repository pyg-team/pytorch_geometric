from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import degree


def normalize_edge_index(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    add_self_loops: bool = True,
    symmetric: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Applies normalization to the edges of a graph.

    This function can add self-loops to the graph and apply either symmetric or
    asymmetric normalization based on the node degrees.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, int], optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        symmetric (bool, optional):  If set to :obj:`True`, symmetric
            normalization (:math:`D^{-1/2} A D^{-1/2}`) is used, otherwise
            asymmetric normalization (:math:`D^{-1} A`).
    """
    if add_self_loops:
        edge_index, _ = add_self_loops_fn(edge_index, num_nodes=num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = degree(row, num_nodes, dtype=torch.get_default_dtype())

    if symmetric:  # D^-1/2 * A * D^-1/2
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    else:  # D^-1 * A
        deg_inv = deg.pow(-1)
        deg_inv[torch.isinf(deg_inv)] = 0
        edge_weight = deg_inv[row]

    return edge_index, edge_weight
