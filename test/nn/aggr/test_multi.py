import pytest
import torch

import torch_geometric.typing
from torch_geometric.testing import withCUDA
from torch_geometric.nn import MultiAggregation

@withCUDA
@pytest.mark.parametrize('aggr_kwargs, expand', [
    (pytest.param(dict(mode='cat'), 3, id='cat')),
    (pytest.param(dict(mode='proj', mode_kwargs=dict(in_channels=16, out_channels=16)), 1, id='proj')),
    (pytest.param(dict(mode='attn', mode_kwargs=dict(in_channels=16, out_channels=16,
                                        num_heads=4)), 1, id='attn')),
    (pytest.param(dict(mode='sum'), 1, id='sum')),
    (pytest.param(dict(mode='mean'), 1, id='mean')),
    (pytest.param(dict(mode='max'), 1, id='max')),
    (pytest.param(dict(mode='min'), 1, id='min')),
    (pytest.param(dict(mode='logsumexp'), 1, id='logsumexp')),
    (pytest.param(dict(mode='std'), 1, id='std')),
    (pytest.param(dict(mode='var'), 1, id='var')),
])
def test_multi_aggr(aggr_kwargs, expand, device):
    # The 'cat' combine mode will expand the output dimensions by
    # the number of aggregators which is 3 here, while the other
    # modes keep output dimensions unchanged.
    x = torch.randn(7, 16, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 3], device=device)
    ptr = torch.tensor([0, 2, 5, 6, 7], device=device)

    aggrs = ['mean', 'sum', 'max']
    aggr = MultiAggregation(aggrs, **aggr_kwargs).to(device)
    aggr.reset_parameters()
    assert str(aggr) == ('MultiAggregation([\n'
                         '  MeanAggregation(),\n'
                         '  SumAggregation(),\n'
                         '  MaxAggregation(),\n'
                         f"], mode={aggr_kwargs['mode']})")

    out = aggr(x, index)
    assert out.size() == (4, expand * x.size(1))

    if not torch_geometric.typing.WITH_TORCH_SCATTER:
        with pytest.raises(ImportError, match="'segment' requires"):
            aggr(x, ptr=ptr)
    else:
        assert torch.allclose(out, aggr(x, ptr=ptr))

    jit = torch.jit.script(aggr)
    assert torch.allclose(out, jit(x, index))

    if torch_geometric.typing.WITH_PT21 and not torch_geometric.typing.WITH_TORCH_SCATTER:
        opt = torch.compile(aggr)
        assert torch.allclose(out, opt(x, index))
