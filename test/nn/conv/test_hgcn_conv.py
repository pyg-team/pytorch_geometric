import torch

from torch_geometric.nn.conv.hgcn_conv import (
    HGCNConv,
    HypAct,
    Hyperboloid,
    HypLinear,
    PoincareBall,
)
from torch_geometric.utils import to_torch_csc_tensor


def test_hgcn_conv_hyperboloid_forward():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = HGCNConv(16, 32, manifold='hyperboloid')
    assert str(conv) == 'HGCNConv(16, 32)'

    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)
    return


def test_hgcn_conv_poincareBall_forward():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = HGCNConv(16, 32, manifold='poincare')
    assert str(conv) == 'HGCNConv(16, 32)'

    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)
    return


def test_hgcn_linear_hyperboloid_forward():
    x = torch.randn(4, 16)
    manifold = Hyperboloid()

    linaer = HypLinear(manifold, 16, 32, 1.0, 0.0, True)

    out1 = linaer(x)
    assert out1.size() == (4, 32)
    return


def test_hgcn_linear_poincareBall_forward():
    x = torch.randn(4, 16)
    manifold = PoincareBall()

    linaer = HypLinear(manifold, 16, 32, 1.0, 0.0, True)

    out1 = linaer(x)
    assert out1.size() == (4, 32)
    return


def test_hypact_hyperboloid_forward():
    x = torch.randn(4, 16)
    manifold = Hyperboloid()

    hypact = HypAct(manifold, 1.0, 1.0, None)

    out1 = hypact(x)
    assert out1.size() == (4, 16)
    return


def test_hypact_poincareBall_forward():
    x = torch.randn(4, 16)
    manifold = PoincareBall()

    hypact = HypAct(manifold, 1.0, 1.0, None)

    out1 = hypact(x)
    assert out1.size() == (4, 16)
    return
