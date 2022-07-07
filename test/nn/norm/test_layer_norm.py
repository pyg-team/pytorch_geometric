import pytest
import torch

from torch_geometric.nn import LayerNorm
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_layer_norm(affine, mode):
    x = torch.randn(100, 16)
    batch = torch.zeros(100, dtype=torch.long)

    norm = LayerNorm(16, affine=affine, mode=mode)
    assert norm.__repr__() == f'LayerNorm(16, mode={mode})'

    if is_full_test():
        torch.jit.script(norm)

    out1 = norm(x)
    assert out1.size() == (100, 16)
    assert torch.allclose(norm(x, batch), out1, atol=1e-6)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100], atol=1e-6)
    assert torch.allclose(out1, out2[100:], atol=1e-6)
