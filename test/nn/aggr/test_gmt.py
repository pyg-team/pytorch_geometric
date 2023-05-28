import torch

from torch_geometric.nn.aggr import GraphMultisetTransformer
from torch_geometric.testing import is_full_test


def test_graph_multiset_transformer():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = GraphMultisetTransformer(16, k=2, heads=2)
    aggr.reset_parameters()
    assert str(aggr) == ('GraphMultisetTransformer(16, k=2, heads=2, '
                         'layer_norm=False, dropout=0.0)')

    out = aggr(x, index)
    assert out.size() == (3, 16)

    if is_full_test():
        jit = torch.jit.script(aggr)
        assert torch.allclose(jit(x, index), out)
