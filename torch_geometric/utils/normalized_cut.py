from torch_geometric.utils import degree, new


def normalized_cut(edge_index, edge_attr, num_nodes=None):
    row, col = edge_index
    deg = 1 / degree(row, num_nodes, new(edge_attr))
    deg = deg[row] + deg[col]
    return edge_attr * deg
