import torch

from torch_geometric.nn import XConv
from torch_geometric.testing import is_full_test, withPackage


@withPackage('torch_cluster')
def test_x_conv():
    x = torch.randn(8, 16)
    pos = torch.rand(8, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    conv = XConv(16, 32, dim=3, kernel_size=2, dilation=2)
    assert str(conv) == 'XConv(16, 32)'

    torch.manual_seed(12345)
    out1 = conv(x, pos)
    assert out1.size() == (8, 32)

    torch.manual_seed(12345)
    out2 = conv(x, pos, batch)
    assert out2.size() == (8, 32)

    if is_full_test():
        jit = torch.jit.script(conv)

        torch.manual_seed(12345)
        assert torch.allclose(jit(x, pos), out1, atol=1e-6)

        torch.manual_seed(12345)
        assert torch.allclose(jit(x, pos, batch), out2, atol=1e-6)
