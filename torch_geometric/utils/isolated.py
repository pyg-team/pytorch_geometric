from torch_unique import unique

from .num_nodes import get_num_nodes
from .loop import remove_self_loops


def contains_isolated_nodes(edge_index, num_nodes=None):
    num_nodes = get_num_nodes(edge_index, num_nodes)
    row, _ = remove_self_loops(edge_index)
    return unique(row).size(0) < num_nodes
