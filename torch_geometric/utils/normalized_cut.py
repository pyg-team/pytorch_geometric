from torch_geometric.utils import degree


def normalized_cut(edge_index, edge_attr, num_nodes=None):
    row, col = edge_index
    deg = 1 / degree(row, num_nodes, edge_attr.dtype, edge_attr.device)
    deg = deg[row] + deg[col]
    cut = edge_attr * deg
    return cut
