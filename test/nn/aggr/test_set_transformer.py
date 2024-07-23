import torch

from torch_geometric.nn.aggr import SetTransformerAggregation
from torch_geometric.testing import is_full_test


def test_set_transformer_aggregation():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 3])

    aggr = SetTransformerAggregation(16, num_seed_points=2, heads=2)
    aggr.reset_parameters()
    assert str(aggr) == ('SetTransformerAggregation(16, num_seed_points=2, '
                         'heads=2, layer_norm=False, dropout=0.0)')

    out = aggr(x, index)
    assert out.size() == (4, 2 * 16)
    assert out.isnan().sum() == 0
    assert out[2].abs().sum() == 0

    if is_full_test():
        jit = torch.jit.script(aggr)
        assert torch.allclose(jit(x, index), out)
