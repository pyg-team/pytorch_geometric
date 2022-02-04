import pytest
import torch

from torch_geometric.nn import GroupNorm


@pytest.mark.parametrize('affine', [True, False])
def test_group_norm(affine):
    x = torch.randn(100, 16)
    batch = torch.zeros(100, dtype=torch.long)

    norm = GroupNorm(16, 4, affine=affine)
    assert norm.__repr__() == ('GroupNorm(in_channels=16,'
                               f' groups=4, affine={affine})')
    torch.jit.script(norm)
    out1 = norm(x)
    assert out1.size() == (100, 16)
    assert torch.allclose(norm(x, batch), out1)
    assert torch.allclose(norm(x), out1)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100])
    assert torch.allclose(out1, out2[100:])
