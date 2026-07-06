import torch

import torch_geometric.typing
from torch_geometric.nn import BLISConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_blis_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = BLISConv(16)
    assert str(conv) == 'BLISConv(16, out_channels=192, K=4, activation=blis)'
    # (K + 2) filters * 2 activations * 16 channels = 6 * 2 * 16 = 192.
    assert conv.out_channels == 192

    out1 = conv(x, edge_index)
    assert out1.size() == (4, 192)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 192)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv(x, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out1, atol=1e-6)
        assert torch.allclose(jit(x, edge_index, value), out2, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj3.t()), out1, atol=1e-6)
            assert torch.allclose(jit(x, adj4.t()), out2, atol=1e-6)


def test_blis_conv_scalar_signal():
    x = torch.randn(4)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = BLISConv(1)
    out = conv(x, edge_index)
    assert out.size() == (4, conv.out_channels) == (4, 12)


def test_blis_conv_identity_activation():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = BLISConv(8, activation='identity')
    out = conv(x, edge_index)
    # No bi-Lipschitz doubling: 6 filters * 8 channels = 48.
    assert conv.out_channels == 48
    assert out.size() == (4, 48)


def test_blis_conv_invalid_activation():
    with_error = False
    try:
        BLISConv(8, activation='relu')
    except ValueError:
        with_error = True
    assert with_error


def test_blis_conv_matches_dense_reference():
    # BLISConv (identity) must equal the dense W2 wavelet transform built from
    # the lazy random walk P = 1/2 (I + A D^-1).
    torch.manual_seed(12345)
    n, f = 7, 3
    adj = (torch.rand(n, n) < 0.4).float()
    adj = ((adj + adj.t()) > 0).float()
    adj.fill_diagonal_(0)
    edge_index = adj.nonzero().t().contiguous()
    x = torch.randn(n, f)

    deg = adj.sum(0)
    p = 0.5 * (torch.eye(n) + adj @ torch.diag(1.0 / deg))

    conv = BLISConv(f, activation='identity')
    powers = [torch.eye(n)]
    for _ in range(conv.num_diffusion):
        powers.append(p @ powers[-1])
    levels = torch.stack(
        [powers[k] @ x for k in range(conv.num_diffusion + 1)])
    ref = torch.einsum('ij,jnf->inf', conv.wavelet_constructor, levels)
    ref = ref.permute(1, 0, 2).reshape(n, -1)

    assert torch.allclose(conv(x, edge_index), ref, atol=1e-5)


def test_blis_conv_trainable():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = BLISConv(8, trainable_laziness=True, trainable_scales=True)
    conv(x, edge_index).sum().backward()

    grads = [p.grad for p in conv.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
