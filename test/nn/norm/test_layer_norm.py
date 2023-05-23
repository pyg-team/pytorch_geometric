import pytest
import torch

from torch_geometric.nn import HeteroLayerNorm, LayerNorm
from torch_geometric.testing import is_full_test, withCUDA


@withCUDA
@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_layer_norm(device, affine, mode):
    x = torch.randn(100, 16, device=device)
    batch = torch.zeros(100, dtype=torch.long, device=device)

    norm = LayerNorm(16, affine=affine, mode=mode).to(device)
    assert str(norm) == f'LayerNorm(16, affine={affine}, mode={mode})'

    if is_full_test():
        torch.jit.script(norm)

    out1 = norm(x)
    assert out1.size() == (100, 16)
    assert torch.allclose(norm(x, batch), out1, atol=1e-6)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100], atol=1e-6)
    assert torch.allclose(out1, out2[100:], atol=1e-6)


@withCUDA
@pytest.mark.parametrize('affine', [False, True])
def test_hetero_layer_norm(device, affine):
    x = torch.randn((100, 16), device=device)
    expected = LayerNorm(16, affine=affine, mode='node').to(device)(x)

    # Test single type:
    type_vec = torch.zeros(100, dtype=torch.long, device=device)
    type_ptr = [0, 100]

    norm = HeteroLayerNorm(16, num_types=1, affine=affine).to(device)
    assert str(norm) == 'HeteroLayerNorm(16, num_types=1)'

    out = norm(x, type_vec)
    assert out.size() == (100, 16)
    assert torch.allclose(out, expected, atol=1e-3)
    assert torch.allclose(norm(out, type_ptr=type_ptr), expected, atol=1e-3)

    mean = out.mean(dim=-1)
    std = out.std(unbiased=False, dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)

    # Test multiple types:
    type_vec = torch.arange(5, device=device)
    type_vec = type_vec.view(-1, 1).repeat(1, 20).view(-1)
    type_ptr = [0, 20, 40, 60, 80, 100]

    norm = HeteroLayerNorm(16, num_types=5, affine=affine).to(device)
    assert str(norm) == 'HeteroLayerNorm(16, num_types=5)'

    out = norm(x, type_vec)
    assert out.size() == (100, 16)
    assert torch.allclose(out, expected, atol=1e-3)
    assert torch.allclose(norm(out, type_ptr=type_ptr), expected, atol=1e-3)

    mean = out.mean(dim=-1)
    std = out.std(unbiased=False, dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
