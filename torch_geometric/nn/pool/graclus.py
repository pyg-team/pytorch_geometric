from typing import Optional

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.index import index2ptr
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes


def graclus(edge_index: Tensor, weight: Optional[Tensor] = None,
            num_nodes: Optional[int] = None):
    r"""A greedy clustering algorithm from the `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
    inderjit/public_papers/multilevel_pami.pdf>`_ paper of picking an unmarked
    vertex and matching it with one of its unmarked neighbors (that maximizes
    its edge weight).
    The GPU algorithm is adapted from the `"A GPU Algorithm for Greedy Graph
    Matching" <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`_
    paper.

    Args:
        edge_index (torch.Tensor): The edge indices.
        weight (torch.Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    if not torch_geometric.typing.WITH_GRACLUS:
        raise ImportError('`graclus` requires `pyg-lib>=0.6.0`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, weight = sort_edge_index(edge_index, weight,
                                         num_nodes=num_nodes)
    rowptr = index2ptr(edge_index[0], num_nodes)
    return torch.ops.pyg.graclus_cluster(rowptr, edge_index[1], weight)
