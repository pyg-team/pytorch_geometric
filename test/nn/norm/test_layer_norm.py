import pytest
import torch

from torch_geometric.nn import HeteroLayerNorm, LayerNorm
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_layer_norm(affine, mode):
    x = torch.randn(100, 16)
    batch = torch.zeros(100, dtype=torch.long)

    norm = LayerNorm(16, affine=affine, mode=mode)
    assert str(norm) == f'LayerNorm(16, affine={affine}, mode={mode})'

    if is_full_test():
        torch.jit.script(norm)

    out1 = norm(x)
    assert out1.size() == (100, 16)
    assert torch.allclose(norm(x, batch), out1, atol=1e-6)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100], atol=1e-6)
    assert torch.allclose(out1, out2[100:], atol=1e-6)


@pytest.mark.parametrize('affine', [False, True])
def test_hetero_layer_norm(affine):
    num_types = 5
    x = torch.randn(100, 16)
    types = torch.randint(num_types, (100, ))

    hetero_norm = HeteroLayerNorm(16, num_types=num_types,
                                  elementwise_affine=affine)
    norm = torch.nn.LayerNorm(16, elementwise_affine=False)

    real_norm = hetero_norm(x, types)

    if affine:
        out = hetero_norm(x, types)
        weight = hetero_norm.weight
        bias = hetero_norm.bias
        node = 5
        node_type = types[node]
        assert torch.allclose(
            out[5], real_norm[5] * weight[node_type] + bias[node_type])
    else:
        out = norm(x)
        assert torch.allclose(out, real_norm, atol=1e-6)
