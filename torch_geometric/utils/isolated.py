import torch

from .num_nodes import maybe_num_nodes
from .loop import remove_self_loops


def contains_isolated_nodes(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    (row, _), _ = remove_self_loops(edge_index)
    return torch.unique(row).size(0) < num_nodes
