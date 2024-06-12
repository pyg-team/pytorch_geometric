from typing import Optional, Union

import torch
import torch.linalg as TLA
from torch import Tensor


def build_adj_dict(num_nodes: int, edge_index: Tensor) -> dict:
    r"""A function to turn a list of edges (edge_index) into an adjacency list,
    stored in a dictionary with vertex numbers as keys and lists of adjacent
    nodes as values.

    Args:
        num_nodes (int): number of nodes
        edge_index (torch.Tensor): edge list

    :rtype: dict
    """
    # initialize adjacency dict with empty neighborhoods for all nodes
    adj_dict: dict = {nodeid: [] for nodeid in range(num_nodes)}

    for eidx in range(edge_index.shape[1]):
        ctail, chead = edge_index[0, eidx].item(), edge_index[1, eidx].item()

        if chead not in adj_dict[ctail]:
            adj_dict[ctail].append(chead)

    return adj_dict


@torch.no_grad()
def dirichlet_energy(
    feat_matrix: Tensor,
    edge_index: Optional[Tensor] = None,
    adj_dict: Optional[dict] = None,
    p: Optional[Union[int, float]] = 2,
) -> float:
    r"""The 'Dirichlet Energy' node similarity measure from the
    `"A Survey on Oversmoothing in Graph Neural Networks"
    <https://arxiv.org/abs/2303.10993>`_ paper.

    .. math::
        \mu\left(\mathbf{X}^n\right) =
        \sqrt{\mathcal{E}\left(\mathbf{X}^n\right)}

    with

    .. math::
        \mathcal{E}(\mathbf{X}^n) = \mathrm{Ave}_{i\in \mathcal{V}}
        \sum_{j \in \mathcal{N}_i} ||\mathbf{X}_{i}^n - \mathbf{X}_{j}^n||_p ^2

    Args:
        feat_matrix (torch.Tensor): The node feature matrix.
        edge_index (torch.Tensor, optional): The edge list
            (default: :obj:`None`)
        adj_dict (dict, optional): The adjacency dictionary
            (default: :obj:`None`)
        p (int or float, optional): The order of the norm (default: :obj:`2`)

    :rtype: float
    """
    num_nodes: int = feat_matrix.shape[0]
    de: Tensor = torch.tensor(0, dtype=torch.float32)

    if adj_dict is None:
        if edge_index is None:
            raise ValueError(
                "Neither 'edge_index' nor 'adj_dict' was provided.")
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index)

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return TLA.vector_norm(x_i - x_js, ord=p, dim=1).square().sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]

        de += inner(own_feat_vector, nbh_feat_matrix).cpu()

    return torch.sqrt(de / num_nodes).item()


@torch.no_grad()
def mean_average_distance(
    feat_matrix: Tensor,
    edge_index: Optional[Tensor] = None,
    adj_dict: Optional[dict] = None,
) -> float:
    r"""The 'Mean Average Distance' node similarity measure from the
    `"A Survey on Oversmoothing in Graph Neural Networks"
    <https://arxiv.org/abs/2303.10993>`_ paper.

    .. math::
        \mu(\mathbf{X}^n) = \mathrm{Ave}_{i\in \mathcal{V}}
        \sum_{j \in \mathcal{N}_i}
        \frac{{\mathbf{X}_i ^n}^\mathrm{T}\mathbf{X}_j ^n}
        {||\mathbf{X}_i ^n|| ||\mathbf{X}_j^n||}

    Args:
        feat_matrix (torch.Tensor): The node feature matrix.
        edge_index (torch.Tensor, optional): The edge list
            (default: :obj:`None`)
        adj_dict (dict, optional): The adjacency dictionary
            (default: :obj:`None`)

    :rtype: float
    """
    num_nodes: int = feat_matrix.shape[0]
    mad: Tensor = torch.tensor(0, dtype=torch.float32)

    if adj_dict is None:
        if edge_index is None:
            raise ValueError(
                "Neither 'edge_index' nor 'adj_dict' was provided.")
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index)

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return (
            1 - (x_i @ x_js.t()) /
            (TLA.vector_norm(x_i, ord=2) * TLA.vector_norm(x_js, ord=2, dim=1))
        ).sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]

        mad += inner(own_feat_vector, nbh_feat_matrix).cpu()

    return (mad / num_nodes).item()
