import torch

from torch_geometric.utils import (
    to_batch_edge_index,
    unbatch,
    unbatch_edge_index,
)


def test_unbatch():
    src = torch.arange(10)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 4])

    out = unbatch(src, batch)
    assert len(out) == 5
    for i in range(len(out)):
        assert torch.equal(out[i], src[batch == i])


def test_unbatch_edge_index():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5],
    ])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    edge_indices = unbatch_edge_index(edge_index, batch)
    assert edge_indices[0].tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
    assert edge_indices[1].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]


def test_to_batch_edge_index():
    # Test basic functionality
    edge_index_list = [
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    ]

    edge_index, batch = to_batch_edge_index(edge_index_list)

    expected_edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5],
    ])
    expected_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    assert torch.equal(edge_index, expected_edge_index)
    assert torch.equal(batch, expected_batch)


def test_to_batch_edge_index_empty_list():
    # Test with empty list
    edge_index, batch = to_batch_edge_index([])
    assert edge_index.shape == (2, 0)
    assert batch.shape == (0, )


def test_to_batch_edge_index_single_graph():
    # Test with single graph
    edge_index_list = [torch.tensor([[0, 1, 2], [1, 2, 0]])]
    edge_index, batch = to_batch_edge_index(edge_index_list)

    assert torch.equal(edge_index, edge_index_list[0])
    assert torch.equal(batch, torch.tensor([0, 0, 0]))


def test_to_batch_edge_index_with_empty_graphs():
    # Test with some empty graphs
    edge_index_list = [
        torch.tensor([[0, 1], [1, 0]]),
        torch.empty((2, 0), dtype=torch.long),
        torch.tensor([[0, 1], [1, 0]]),
    ]

    edge_index, batch = to_batch_edge_index(edge_index_list)

    expected_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    expected_batch = torch.tensor([0, 0, 2, 2])

    assert torch.equal(edge_index, expected_edge_index)
    assert torch.equal(batch, expected_batch)


def test_to_batch_edge_index_roundtrip():
    # Test that to_batch_edge_index and unbatch_edge_index are inverses
    edge_index_list = [
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
    ]

    edge_index, batch = to_batch_edge_index(edge_index_list)
    unbatched = unbatch_edge_index(edge_index, batch)

    assert len(unbatched) == len(edge_index_list)
    for i, (original, recovered) in enumerate(zip(edge_index_list, unbatched)):
        assert torch.equal(original, recovered), f"Mismatch at index {i}"


def test_to_batch_edge_index_different_sizes():
    # Test with graphs of different sizes
    edge_index_list = [
        torch.tensor([[0], [0]]),  # Self-loop
        torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),  # 5-node cycle
        torch.tensor([[0, 1], [1, 0]]),  # 2-node bidirectional
    ]

    edge_index, batch = to_batch_edge_index(edge_index_list)

    # Verify the batch vector has correct size
    assert batch.shape[0] == 1 + 5 + 2  # Total nodes

    # Verify edge_index has correct number of edges
    assert edge_index.shape[1] == 1 + 5 + 2  # Total edges

    # Verify roundtrip
    unbatched = unbatch_edge_index(edge_index, batch)
    assert len(unbatched) == len(edge_index_list)
    for original, recovered in zip(edge_index_list, unbatched):
        assert torch.equal(original, recovered)
