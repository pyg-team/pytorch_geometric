import torch

from torch_geometric.nn.aggr.lcm import LCMAggregation


def test_lcm_aggregation1():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = LCMAggregation(16, 32)
    assert str(aggr) == 'LCMAggregation(16, 32, project=True)'

    out = aggr(x, index)
    assert out.size() == (3, 32)


def test_lcm_aggregation2():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = LCMAggregation(16, 16, project=False)
    assert str(aggr) == 'LCMAggregation(16, 16, project=False)'

    out = aggr(x, index)
    assert out.size() == (3, 16)


def test_lcm_aggregation3():
    try:
        LCMAggregation(16, 32, project=False)
    except ValueError:
        return

    assert False
