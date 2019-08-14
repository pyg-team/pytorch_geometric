import torch
import random
import numpy as np


def negative_sampling(pos_edge_index, num_nodes, max_num_samples=None):
    r"""Sample negative edges of a graph ({0, ..., num_nodes - 1}, pos_edge_index).

    Args:
        pos_edge_index (LongTensor): Existing adjacency list of the given graph.
        num_nodes (int): Number of nodes in the given graph.
        max_num_samples (int, optional): Maximum number of negative samples. The
            number of negative samples that this method returns cannot exceed max_num_samples.

    :rtype: LongTensor
    """

    max_num_samples = max_num_samples or pos_edge_index.size(1)

    # Handle '2*|pos_edges| > num_nodes^2' case.
    num_samples = min(max_num_samples, num_nodes * num_nodes - pos_edge_index.size(1))

    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes ** 2)
    perm = torch.tensor(random.sample(rng, num_samples))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(pos_edge_index.device)


def negative_triangle_sampling(edge_index, num_nodes):
    r"""Sample negative edges of a graph ({0, ..., num_nodes - 1}, edge_index),
    with the form of triangle. For return tuple (i, j, k), [i, j] is existing adjacency
    list of the given graph, [k, i] is the  adjacency list of sampled negative edges.

    Args:
        edge_index (LongTensor): Existing adjacency list of the given graph.
        num_nodes (int): Number of nodes in the given graph.

    :rtype: Tuple[LongTensor]
    """
    device = edge_index.device
    i, j = edge_index.to(torch.device('cpu'))
    idx_1 = i * num_nodes + j
    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = k * num_nodes + i
    mask = torch.from_numpy(np.isin(idx_2, idx_1).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        idx_2 = tmp * num_nodes + i[rest]
        mask = torch.from_numpy(np.isin(idx_2, idx_1).astype(np.uint8))
        k[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]
    return i.to(device), j.to(device), k.to(device)
