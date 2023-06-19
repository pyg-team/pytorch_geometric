import torch


def remove_single_hyperedge(edge_mask, edge_index):
    _edge_index = edge_index[:, edge_mask]
    unique_elements, counts = torch.unique(_edge_index[1], return_counts=True)
    _edge_index = [[], []]
    single_hyperedge = unique_elements[counts == 1]

    for i in range(edge_index.size(dim=1)):
        if edge_index[1][i] not in single_hyperedge:
            if edge_mask[i] is True:
                _edge_index[0].append(edge_index[0][i])
                _edge_index[1].append(edge_index[1][i])
        elif edge_mask[i] is True:
            edge_mask[i] = False

    return edge_mask, torch.tensor(_edge_index)
