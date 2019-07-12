import torch
from torch_geometric.nn.pool import edge_pool
from torch_scatter import scatter_add


def test_compute_new_edges():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    old_to_new_node_idx = torch.tensor([0, 2, 0, 3, 1, 1])
    new_edge_index = edge_pool._compute_new_edges(
        old_to_new_node_idx,
        edge_index,
        n_nodes=4
    )

    goal_edge_index = torch.tensor([[0, 0, 0, 1, 2, 2, 3, 3],
                                    [0, 2, 3, 1, 0, 3, 0, 2]])
    assert torch.all(new_edge_index == goal_edge_index)


def test_compute_edge_score_softmax():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw_edge_scores = torch.randn(edge_index.shape[1])
    edge_score = edge_pool.EdgePooling.compute_edge_score_softmax(
        raw_edge_scores, edge_index)
    # test validity of all results (0 < edge_score < 1)
    assert torch.all(0 < edge_score)
    assert torch.all(edge_score <= 1)

    # test whether all incoming edge scores sum up to 1
    edge_score_sum = scatter_add(edge_score, edge_index[1])
    assert torch.allclose(edge_score_sum, torch.ones_like(edge_score_sum))

    # test whether the edges that have no alternatives have a score of 1
    assert edge_score_sum[-1] == 1
    assert edge_score_sum[-2] == 1


def test_compute_edge_score_tanh():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw_edge_scores = torch.randn(edge_index.shape[1])
    edge_score = edge_pool.EdgePooling.compute_edge_score_tanh(
        raw_edge_scores, edge_index)
    # test validity of all results (-1 < edge_score < 1)
    assert torch.all(-1 < edge_score)
    assert torch.all(edge_score <= 1)

    # test order-stability
    assert torch.all(
        torch.argsort(raw_edge_scores) == torch.argsort(edge_score)
    )


def test_compute_edge_score_sigmoid():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw_edge_scores = torch.randn(edge_index.shape[1])
    edge_score = edge_pool.EdgePooling.compute_edge_score_sigmoid(
        raw_edge_scores, edge_index)
    # test validity of all results (0 < edge_score < 1)
    assert torch.all(0 < edge_score)
    assert torch.all(edge_score <= 1)

    # test order-stability
    assert torch.all(
        torch.argsort(raw_edge_scores) == torch.argsort(edge_score)
    )


def test_edgepooling():
    x = torch.tensor([[0], [1], [2], [3], [4], [5], [-1]], dtype=torch.float)
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 0]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 0])
    ep = edge_pool.EdgePooling(in_channels=1)

    # setting parameters fixed so we can test the expected outcome
    ep.net.weight[0, 0] = 1
    ep.net.weight[0, 1] = 1
    ep.net.bias[0] = 0

    new_x, new_edge_index, new_batch, unpool_info = ep(
        x, edge_index, batch, return_unpool_info=True
    )

    assert new_x.shape[0] == new_batch.shape[0] == 4
    assert torch.all(new_batch == torch.tensor([1, 0, 0, 0]))
    assert torch.all(new_edge_index == torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [0, 1, 2, 1, 2, 2]
    ]))

    assert ep.__repr__() == 'EdgePooling(compute_edge_score_softmax, ' \
        '1, dropout=0, add_to_edge_score=0.5)'

    # testing unpooling
    unpooled_x, unpooled_edge_index, unpooled_batch = ep.unpool(
        new_x, unpool_info
    )
    assert torch.all(unpooled_edge_index == edge_index)
    assert torch.all(unpooled_batch == batch)
    assert x.shape == unpooled_x.shape
    assert torch.allclose(unpooled_x, torch.tensor([
        [1], [1], [5], [5], [9], [9], [-1]
    ], dtype=torch.float))
