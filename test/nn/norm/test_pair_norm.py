import pytest
import torch

from torch_geometric.nn import PairNorm
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('scale_individually', [False, True])
def test_pair_norm(scale_individually):
    x = torch.randn(100, 16)
    batch = torch.zeros(100, dtype=torch.long)

    norm = PairNorm(scale_individually=scale_individually)
    assert str(norm) == 'PairNorm()'

    if is_full_test():
        torch.jit.script(norm)

    out1 = norm(x)
    assert out1.size() == (100, 16)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100], atol=1e-6)
    assert torch.allclose(out1, out2[100:], atol=1e-6)
