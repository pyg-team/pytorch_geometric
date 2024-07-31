import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops, degree
from typing import Tuple


def normalize_edge_index(
        edge_index: Tensor,
        num_nodes: int,
        self_loop: bool = True,
        symmetric_normalization: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Normalizes the edge index of a graph for use in GNNs.

    This function can add self-loops to the graph and apply either symmetric or
    asymmetric normalization based on the node degrees.

    Parameters:
    -----------
    edge_index : Tensor
        The edge index tensor of shape [2, num_edges] representing the graph.
        The first row contains the source nodes, and the second row contains the target nodes.
    num_nodes : int
        The number of nodes in the graph.
    self_loop : bool, optional (default=True)
        If True, self-loops are added to the graph.
    symmetric_normalization : bool, optional (default=True)
        If True, symmetric normalization (D^-1/2 A D^-1/2) is applied. Otherwise,
        asymmetric normalization (D^-1 A) is used.

    Returns:
    --------
    Tuple[Tensor, Tensor]
        A tuple containing the edge index and edge weights:
        - edge_index: The possibly modified edge index with self-loops.
        - edge_weight: The normalized edge weights.
    """

    if self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index[0], edge_index[1]

    deg = degree(row, num_nodes, dtype=torch.float)

    if symmetric_normalization:
        # Symmetric normalization: D^-1/2 * A * D^-1/2
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    else:
        # Asymmetric normalization: D^-1 * A
        deg_inv = deg.pow(-1)
        deg_inv[torch.isinf(deg_inv)] = 0
        edge_weight = deg_inv[row]

    return edge_index, edge_weight
