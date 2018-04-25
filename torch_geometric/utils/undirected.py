import torch
from torch_geometric.utils import coalesce


def to_undirected(edge_index, num_nodes=None):
    row, col = edge_index

    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = coalesce(edge_index, num_nodes)

    return edge_index
