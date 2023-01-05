import torch

from torch_geometric.nn import DeepSetsAggregation, Linear


def test_deep_sets_aggregation():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = DeepSetsAggregation(
        local_nn=Linear(16, 32),
        global_nn=Linear(32, 64),
    )
    aggr.reset_parameters()
    assert str(aggr) == ('DeepSetsAggregation('
                         'local_nn=Linear(16, 32, bias=True), '
                         'global_nn=Linear(32, 64, bias=True))')

    out = aggr(x, index)
    assert out.size() == (3, 64)
