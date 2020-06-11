import torch
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling,
                                   batched_negative_sampling, to_undirected,
                                   is_undirected)


def test_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_sampling(edge_index)
    assert neg_edge_index.size(1) <= edge_index.size(1)

    adj = torch.zeros(4, 4, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    neg_adj = torch.zeros(4, 4, dtype=torch.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True
    assert (adj & neg_adj).sum() == 0

    neg_edge_index = negative_sampling(edge_index, num_neg_samples=2)
    assert neg_edge_index.size(1) <= 2

    edge_index = to_undirected(edge_index)
    neg_edge_index = negative_sampling(edge_index, force_undirected=True)
    assert is_undirected(neg_edge_index)
    assert neg_edge_index.size(1) <= edge_index.size(1)

    adj = torch.zeros(4, 4, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    neg_adj = torch.zeros(4, 4, dtype=torch.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True
    assert (adj & neg_adj).sum() == 0


def test_structured_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    i, j, k = structured_negative_sampling(edge_index)
    assert i.size(0) == edge_index.size(1)
    assert j.size(0) == edge_index.size(1)
    assert k.size(0) == edge_index.size(1)

    adj = torch.zeros(4, 4, dtype=torch.bool)
    adj[i, j] = 1

    neg_adj = torch.zeros(4, 4, dtype=torch.bool)
    neg_adj[i, k] = 1
    assert (adj & neg_adj).sum() == 0


def test_batched_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    neg_edge_index = batched_negative_sampling(edge_index, batch)
    assert neg_edge_index.size(1) <= edge_index.size(1)

    adj = torch.zeros(8, 8, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    neg_adj = torch.zeros(8, 8, dtype=torch.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True
    assert (adj & neg_adj).sum() == 0
    assert neg_adj[:4, 4:].sum() == 0
    assert neg_adj[4:, :4].sum() == 0
