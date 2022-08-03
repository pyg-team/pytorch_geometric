import torch

from torch_geometric.nn.aggr import SortAggregation


def test_sort_aggregation():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    index = torch.tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])

    aggr = SortAggregation(k=5)
    assert str(aggr) == 'SortAggregation(k=5)'

    out = aggr(x, index)
    assert out.size() == (2, 5 * 4)

    out_dim = out = aggr(x, index, dim=0)
    assert torch.allclose(out_dim, out)

    out = out.view(2, 5, 4)

    # First graph output has been filled up with zeros.
    assert out[0, -1].tolist() == [0, 0, 0, 0]

    # Nodes are sorted.
    expected = 3 - torch.arange(4)
    assert out[0, :4, -1].argsort().tolist() == expected.tolist()

    expected = 4 - torch.arange(5)
    assert out[1, :, -1].argsort().tolist() == expected.tolist()


def test_sort_aggregation_smaller_than_k():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    index = torch.tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])

    # Set k which is bigger than both N_1=4 and N_2=6.
    aggr = SortAggregation(k=10)
    assert str(aggr) == 'SortAggregation(k=10)'

    out = aggr(x, index)
    assert out.size() == (2, 10 * 4)

    out_dim = out = aggr(x, index, dim=0)
    assert torch.allclose(out_dim, out)

    out = out.view(2, 10, 4)

    # Both graph outputs have been filled up with zeros.
    assert out[0, -1].tolist() == [0, 0, 0, 0]
    assert out[1, -1].tolist() == [0, 0, 0, 0]

    # Nodes are sorted.
    expected = 3 - torch.arange(4)
    assert out[0, :4, -1].argsort().tolist() == expected.tolist()

    expected = 5 - torch.arange(6)
    assert out[1, :6, -1].argsort().tolist() == expected.tolist()


def test_sort_aggregation_dim_size():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    index = torch.tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])

    aggr = SortAggregation(k=5)
    assert str(aggr) == 'SortAggregation(k=5)'

    # expand batch output by 1
    out = aggr(x, index, dim_size=3)
    assert out.size() == (3, 5 * 4)

    out = out.view(3, 5, 4)

    # Both first and last graph outputs have been filled up with zeros.
    assert out[0, -1].tolist() == [0, 0, 0, 0]
    assert out[2, -1].tolist() == [0, 0, 0, 0]
