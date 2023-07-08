import torch


def remove_single_hyperedge(edge_mask, edge_index):
    r"""Takes a Hypergraph's edge_index, and remove the edges that only
     connect one node. Used when computing a subgraph.

     Examples:
        >>> edge_index = torch.tensor([
        ...     [0, 1, 0, 2, 1, 1, 2, 3],
        ...     [0, 0, 1, 1, 1, 2, 2, 2]
        >>> ])
        >>> edge_mask = torch.tensor([
        ... False, True, True, True, False, True, True, True
        ... ])
        >>> edge_mask, edge_index = remove_single_hyperedge(
        ... edge_mask, edge_index
        ... )
        >>> edge_index
        tensor([[0, 2, 1, 2, 3],
        [1, 1, 2, 2, 2]])

    .. note::

        This method is for internal use only.
    Args:
        edge_mask (BoolTensor): The edges to keep.
        edge_index (LongTensor or BoolTensor): The original hyperedges.
    """
    _edge_index = edge_index[:, edge_mask]
    unique_elements, counts = torch.unique(_edge_index[1], return_counts=True)
    _edge_index = [[], []]
    single_hyperedge = unique_elements[counts == 1]

    for i in range(edge_index.size(dim=1)):
        if edge_index[1][i] not in single_hyperedge:
            if edge_mask[i]:
                _edge_index[0].append(edge_index[0][i])
                _edge_index[1].append(edge_index[1][i])
        elif edge_mask[i]:
            edge_mask[i] = False

    return edge_mask, torch.tensor(_edge_index)
