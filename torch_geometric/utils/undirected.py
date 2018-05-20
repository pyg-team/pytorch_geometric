import torch

from .coalesce import coalesce


def is_undirected(edge_index, num_nodes=None):
    edge_index, _ = coalesce(edge_index, num_nodes=num_nodes)
    undirected_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index.size(1) == undirected_edge_index.size(1)


def to_undirected(edge_index, num_nodes=None):
    row, col = edge_index

    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index
