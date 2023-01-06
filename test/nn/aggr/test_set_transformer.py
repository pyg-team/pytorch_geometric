import torch

from torch_geometric.nn.aggr import SetTransformerAggregation
from torch_geometric.testing import is_full_test


def test_set_transformer_aggregation():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = SetTransformerAggregation(16, num_seed_points=2, heads=2)
    aggr.reset_parameters()
    assert str(aggr) == ('SetTransformerAggregation(16, num_seed_points=2, '
                         'heads=2, layer_norm=False)')

    out = aggr(x, index)
    assert out.size() == (3, 2 * 16)

    if is_full_test():
        jit = torch.jit.script(aggr)
        assert torch.allclose(jit(x, index), out)
