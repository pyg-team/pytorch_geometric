import torch


def has_self_loops(edge_index):
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0


def add_self_loops(edge_index, num_nodes=None):
    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes

    loop = torch.arange(num_nodes, out=edge_index.new())
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)

    return edge_index


def remove_self_loops(edge_index):
    row, col = edge_index
    mask = row != col
    mask = mask.unsqueeze(0).expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index
