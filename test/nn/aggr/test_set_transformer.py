import warnings

import torch

import torch_geometric.typing
from torch_geometric.nn.aggr import SetTransformerAggregation
from torch_geometric.testing import is_full_test, withCUDA


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
    if torch_geometric.typing.WITH_PT25:
        if not out[2].abs().sum() != 0:
            warnings.warn("'SetTransformerAggregation' broken on PyTorch>2.4",
                          stacklevel=2)
    else:
        assert out[2].abs().sum() == 0

    if is_full_test():
        jit = torch.jit.script(aggr)
        assert torch.allclose(jit(x, index), out)


@withCUDA
def test_set_transformer_aggregation_mha_fastpath(device, monkeypatch):
    # Record whether the MHA fastpath is enabled at every attention call:
    calls = []
    forward = torch.nn.MultiheadAttention.forward

    def spy(self, query, key, value, key_padding_mask=None, *args, **kwargs):
        has_mask = key_padding_mask is not None
        calls.append((has_mask, torch.backends.mha.get_fastpath_enabled()))
        return forward(self, query, key, value, key_padding_mask, *args,
                       **kwargs)

    monkeypatch.setattr(torch.nn.MultiheadAttention, 'forward', spy)

    x = torch.randn(6, 16, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 3], device=device)

    aggr = SetTransformerAggregation(16, num_seed_points=2, heads=2)
    aggr = aggr.to(device)

    assert torch.backends.mha.get_fastpath_enabled()

    aggr.eval()
    with torch.no_grad():
        out = aggr(x, index)
    assert out.size() == (4, 2 * 16)
    assert out.isfinite().all()

    # The fastpath is only disabled for masked attention on CUDA in eval mode:
    masked = [fastpath for has_mask, fastpath in calls if has_mask]
    unmasked = [fastpath for has_mask, fastpath in calls if not has_mask]
    assert len(masked) > 0 and len(unmasked) > 0
    if device.type == 'cuda':
        assert not any(masked)
    else:
        assert all(masked)
    assert all(unmasked)
    assert torch.backends.mha.get_fastpath_enabled()

    # The fastpath is left untouched in training mode:
    calls.clear()
    aggr.train()
    out = aggr(x, index)
    assert out.size() == (4, 2 * 16)
    assert len(calls) > 0
    assert all(fastpath for _, fastpath in calls)
    assert torch.backends.mha.get_fastpath_enabled()
