import numpy as np
import torch
from torch_geometric.utils import negative_sampling, structured_negative_sampling, batched_negative_sampling, subgraph


def test_negative_sampling():
    pos_edge_index = torch.as_tensor([[0, 0, 0, 0, 1, 2],
                                      [0, 1, 2, 3, 2, 3]])
    num_nodes = 4

    neg_edge_index = negative_sampling(pos_edge_index, num_nodes)
    assert neg_edge_index.size(1) <= pos_edge_index.size(1)

    pos_idx = pos_edge_index[0] * num_nodes + pos_edge_index[1]
    sampled_idx = neg_edge_index[0] * num_nodes + neg_edge_index[1]
    mask = torch.from_numpy(np.isin(pos_idx, sampled_idx).astype(np.uint8))
    assert mask.sum() == 0

    num_neg_samples = 3
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes, num_neg_samples=num_neg_samples)
    assert neg_edge_index.size(1) <= num_neg_samples

    pos_edge_index = torch.as_tensor([[0, 0, 1],
                                      [0, 1, 0]])
    num_nodes = 2
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes)
    assert neg_edge_index.size(1) == 1


def test_structured_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 0, 0, 1, 2],
                                  [0, 1, 2, 3, 2, 3]])
    num_nodes = 4

    i, j, k = structured_negative_sampling(edge_index, num_nodes)
    assert i.size(0) == edge_index.size(1)

    ij_idx = i * num_nodes + j
    ki_idx = k * num_nodes + i
    mask = torch.from_numpy(np.isin(ij_idx, ki_idx).astype(np.uint8))
    assert mask.sum() == 0


def test_batched_negative_sampling():
    one_edge_index = torch.as_tensor([[0, 0, 0, 0],
                                      [0, 1, 2, 3]])

    edge_index = torch.cat([one_edge_index, one_edge_index + 4], dim=1)
    num_nodes = 8
    batch = torch.as_tensor([0 for _ in range(num_nodes // 2)] + [1 for _ in range(num_nodes // 2)]).long()

    neg_edge_index = batched_negative_sampling(edge_index, batch=batch)
    assert neg_edge_index.size(1) <= edge_index.size(1)

    neg_edge_index_list = [subgraph(torch.as_tensor(nodes), neg_edge_index, relabel_nodes=True)[0]
                           for nodes in [[0, 1, 2, 3], [4, 5, 6, 7]]]

    for nei in neg_edge_index_list:
        pos_idx = one_edge_index[0] * 4 + one_edge_index[1]
        sampled_idx = nei[0] * 4 + nei[1]
        mask = torch.from_numpy(np.isin(pos_idx, sampled_idx).astype(np.uint8))
        assert mask.sum() == 0
