from torch_cluster import graclus_cluster


def graclus(edge_index, weight=None):
    row, col = edge_index
    return graclus_cluster(row, col, weight)
