import torch

from torch_geometric.utils import (
    batched_negative_sampling,
    contains_self_loops,
    erdos_renyi_graph,
    is_undirected,
    negative_sampling,
    structured_negative_sampling,
    structured_negative_sampling_feasible,
    to_undirected,
)
from torch_geometric.utils._negative_sampling import (
    edge_index_to_vector_id,
    vector_id_to_edge_index,
)


def is_negative(edge_index, neg_edge_index, size, bipartite):
    adj = torch.zeros(size, dtype=torch.bool)
    neg_adj = torch.zeros(size, dtype=torch.bool)

    adj[edge_index[0], edge_index[1]] = True
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    if not bipartite:
        arange = torch.arange(size[0])
        assert neg_adj[arange, arange].sum() == 0

    return (adj & neg_adj).sum() == 0


def test_edge_index_to_vector_and_vice_versa():
    # Create a fully-connected graph:
    N1, N2 = 13, 17
    row = torch.arange(N1).view(-1, 1).repeat(1, N2).view(-1)
    col = torch.arange(N2).view(1, -1).repeat(N1, 1).view(-1)
    edge_index = torch.stack([row, col], dim=0)

    idx = edge_index_to_vector_id(edge_index, (N1, N2))
    assert idx.tolist() == list(range(N1 * N2))
    edge_index2 = torch.stack(vector_id_to_edge_index(idx, (N1, N2)), dim=0)
    assert edge_index.tolist() == edge_index2.tolist()

    vector_id = torch.arange(N1 * N2)
    edge_index3 = torch.stack(vector_id_to_edge_index(vector_id, (N1, N2)),
                              dim=0)
    assert edge_index.tolist() == edge_index3.tolist()


def test_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_sampling(edge_index)
    assert neg_edge_index.size(1) == edge_index.size(1)
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    neg_edge_index = negative_sampling(edge_index, num_neg_samples=2)
    assert neg_edge_index.size(1) == 2
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    edge_index = to_undirected(edge_index)
    neg_edge_index = negative_sampling(edge_index, force_undirected=True)
    assert neg_edge_index.size(1) == edge_index.size(1) - 1
    assert is_undirected(neg_edge_index)
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)


def test_bipartite_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_sampling(edge_index, num_nodes=(3, 4))
    assert neg_edge_index.size(1) == edge_index.size(1)
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)

    neg_edge_index = negative_sampling(edge_index, num_nodes=(3, 4),
                                       num_neg_samples=2)
    assert neg_edge_index.size(1) == 2
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)


def test_negative_sampling_with_different_edge_density():
    for num_nodes in [10, 100, 1000]:
        for p in [0.1, 0.3, 0.5, 0.8]:
            for is_directed in [False, True]:
                edge_index = erdos_renyi_graph(num_nodes, p, is_directed)
                neg_edge_index = negative_sampling(
                    edge_index, num_nodes, force_undirected=not is_directed)
                assert is_negative(edge_index, neg_edge_index,
                                   (num_nodes, num_nodes), bipartite=False)


def test_bipartite_negative_sampling_with_different_edge_density():
    for num_nodes in [10, 100, 1000]:
        for p in [0.1, 0.3, 0.5, 0.8]:
            size = (num_nodes, int(num_nodes * 1.2))
            n_edges = int(p * size[0] * size[1])
            row, col = torch.randint(size[0], (n_edges, )), torch.randint(
                size[1], (n_edges, ))
            edge_index = torch.stack([row, col], dim=0)
            neg_edge_index = negative_sampling(edge_index, size)
            assert is_negative(edge_index, neg_edge_index, size,
                               bipartite=True)


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
    assert (adj | neg_adj).sum() == edge_index.size(1) + neg_edge_index.size(1)
    assert neg_adj[:4, 4:].sum() == 0
    assert neg_adj[4:, :4].sum() == 0


def test_bipartite_batched_negative_sampling():
    edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
    edge_index2 = edge_index1 + torch.tensor([[2], [4]])
    edge_index3 = edge_index2 + torch.tensor([[2], [4]])
    edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
    src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
    dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    neg_edge_index = batched_negative_sampling(edge_index,
                                               (src_batch, dst_batch))
    assert neg_edge_index.size(1) <= edge_index.size(1)

    adj = torch.zeros(6, 12, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    neg_adj = torch.zeros(6, 12, dtype=torch.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    assert (adj & neg_adj).sum() == 0
    assert (adj | neg_adj).sum() == edge_index.size(1) + neg_edge_index.size(1)


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

    # Test with no self-loops:
    #edge_index = torch.LongTensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 1]])
    i, j, k = structured_negative_sampling(edge_index, num_nodes=4,
                                           contains_neg_self_loops=False)
    neg_edge_index = torch.vstack([i, k])
    assert not contains_self_loops(neg_edge_index)


def test_structured_negative_sampling_sparse():
    num_nodes = 1000
    edge_index = erdos_renyi_graph(num_nodes, 0.1)

    i, j, k = structured_negative_sampling(edge_index, num_nodes=num_nodes,
                                           contains_neg_self_loops=True)
    assert i.size(0) == edge_index.size(1)
    assert j.size(0) == edge_index.size(1)
    assert k.size(0) == edge_index.size(1)

    assert torch.all(torch.ne(k, -1))
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    adj[i, j] = 1

    neg_adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    neg_adj[i, k] = 1
    assert (adj & neg_adj).sum() == 0

    i, j, k = structured_negative_sampling(edge_index, num_nodes=num_nodes,
                                           contains_neg_self_loops=False)
    assert i.size(0) == edge_index.size(1)
    assert j.size(0) == edge_index.size(1)
    assert k.size(0) == edge_index.size(1)

    assert torch.all(torch.ne(k, -1))
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    adj[i, j] = 1

    neg_adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    neg_adj[i, k] = 1
    assert (adj & neg_adj).sum() == 0

    neg_edge_index = torch.vstack([i, k])
    assert not contains_self_loops(neg_edge_index)


def test_structured_negative_sampling_feasible():
    def create_ring_graph(num_nodes):
        forward_edges = torch.stack([
            torch.arange(0, num_nodes, dtype=torch.long),
            (torch.arange(0, num_nodes, dtype=torch.long) + 1) % num_nodes
        ], dim=0)
        backward_edges = torch.stack([
            torch.arange(0, num_nodes, dtype=torch.long),
            (torch.arange(0, num_nodes, dtype=torch.long) - 1) % num_nodes
        ], dim=0)
        return torch.concat([forward_edges, backward_edges], dim=1)

    # ring 3 is always unfeasible
    ring_3 = create_ring_graph(3)
    assert not structured_negative_sampling_feasible(ring_3, 3, False)
    assert not structured_negative_sampling_feasible(ring_3, 3, True)

    # ring 4 is feasible only if we consider self loops
    ring_4 = create_ring_graph(4)
    assert not structured_negative_sampling_feasible(ring_4, 4, False)
    assert structured_negative_sampling_feasible(ring_4, 4, True)

    # ring 5 is always feasible
    ring_5 = create_ring_graph(5)
    assert structured_negative_sampling_feasible(ring_5, 5, False)
    assert structured_negative_sampling_feasible(ring_5, 5, True)
