import torch

from torch_geometric.nn import GravNetConv
from torch_geometric.testing import is_full_test, withPackage


@withPackage('torch_cluster')
def test_gravnet_conv():
    x1 = torch.randn(8, 16)
    x2 = torch.randn(4, 16)
    batch1 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch2 = torch.tensor([0, 0, 1, 1])

    conv = GravNetConv(16, 32, space_dimensions=4, propagate_dimensions=8, k=2)
    assert str(conv) == 'GravNetConv(16, 32, k=2)'

    out11 = conv(x1)
    assert out11.size() == (8, 32)

    out12 = conv(x1, batch1)
    assert out12.size() == (8, 32)

    out21 = conv((x1, x2))
    assert out21.size() == (4, 32)

    out22 = conv((x1, x2), (batch1, batch2))
    assert out22.size() == (4, 32)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1), out11)
        assert torch.allclose(jit(x1, batch1), out12)

        assert torch.allclose(jit((x1, x2)), out21)
        assert torch.allclose(jit((x1, x2), (batch1, batch2)), out22)
