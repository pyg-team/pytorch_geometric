import numpy as np
import torch
from torch_geometric.utils import negative_sampling, negative_triangle_sampling


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

    max_num_samples = 3
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes, max_num_samples=max_num_samples)
    assert neg_edge_index.size(1) <= max_num_samples

    pos_edge_index = torch.as_tensor([[0, 0, 1],
                                      [0, 1, 0]])
    num_nodes = 2
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes)
    assert neg_edge_index.size(1) == 1


def test_negative_triangle_sampling():
    edge_index = torch.as_tensor([[0, 0, 0, 0, 1, 2],
                                  [0, 1, 2, 3, 2, 3]])
    num_nodes = 4

    i, j, k = negative_triangle_sampling(edge_index, num_nodes)
    assert i.size(0) == edge_index.size(1)

    ij_idx = i * num_nodes + j
    ki_idx = k * num_nodes + i
    mask = torch.from_numpy(np.isin(ij_idx, ki_idx).astype(np.uint8))
    assert mask.sum() == 0
