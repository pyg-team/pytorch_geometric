import torch
import random
import numpy as np

from torch_geometric.utils import subgraph


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


def batch_negative_sampling(pos_edge_index, num_nodes=None, batch=None, max_num_samples=None):
    r"""Apply negative sampling to batches of multiple graphs.

    Args:
        pos_edge_index (LongTensor): Existing adjacency list of the given graph.
        num_nodes (int): Number of nodes in the given graph.
        max_num_samples (int, optional): Maximum number of negative samples. The
            number of negative samples that this method returns cannot exceed max_num_samples.
        batch (LongTensor, optional): The Batch object.
    """
    assert (num_nodes is not None) or (batch is not None), "Either num_nodes or batch must not be None"

    if batch is not None:
        nodes_list = [torch.nonzero(batch == b).squeeze() for b in range(batch.max() + 1)]
        num_nodes_list = [len(nodes) for nodes in nodes_list]
        pos_edge_index_list = [subgraph(torch.as_tensor(nodes), pos_edge_index, relabel_nodes=True)[0]
                               for nodes in nodes_list]
    else:
        num_nodes_list = [num_nodes]
        pos_edge_index_list = [pos_edge_index]

    neg_edges_index_list = []
    prev_node_idx = 0
    for num_nodes, pos_edge_index_of_one_graph in zip(num_nodes_list, pos_edge_index_list):
        neg_edges_index = negative_sampling(pos_edge_index_of_one_graph,
                                            num_nodes=num_nodes,
                                            max_num_samples=max_num_samples)
        neg_edges_index += prev_node_idx
        neg_edges_index_list.append(neg_edges_index)
        prev_node_idx += num_nodes
    neg_edges_index = torch.cat(neg_edges_index_list, dim=1)
    return neg_edges_index
