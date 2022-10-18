import torch

from torch_geometric.nn import WLConvContinuous
from torch_geometric.testing import is_full_test


def test_wl_conv():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    conv = WLConvContinuous()
    assert str(conv) == 'WLConvContinuous()'

    out = conv(x, edge_index)
    assert out.tolist() == [[-0.5], [0.0], [0.5]]

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out)

    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))

    out = conv((x1, None), edge_index, edge_weight, size=(4, 2))
    assert out.size() == (2, 8)

    out = conv((x1, x2), edge_index, edge_weight)
    assert out.size() == (2, 8)
