import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import StableChebConv
from torch_geometric.testing import is_full_test


def test_stable_cheb_conv():
    in_channels, out_channels = (16, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = StableChebConv(in_channels, out_channels, K=3)
    assert str(conv) == (f'StableChebConv({in_channels}, {out_channels}, K=3, '
                         f'epsilon=0.5, gamma=0.1, normalization=\'sym\')')

    out1 = conv(x, edge_index)
    assert out1.size() == (num_nodes, out_channels)

    out2 = conv(x, edge_index, edge_weight)
    assert out2.size() == (num_nodes, out_channels)

    out3 = conv(x, edge_index, edge_weight, lambda_max=3.0)
    assert out3.size() == (num_nodes, out_channels)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out1)
        assert torch.allclose(jit(x, edge_index, edge_weight), out2)
        assert torch.allclose(
            jit(x, edge_index, edge_weight, lambda_max=torch.tensor(3.0)),
            out3,
        )


def test_stable_cheb_conv_batch():
    in_channels, out_channels = (8, 8)

    x1 = torch.randn(4, in_channels)
    edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight1 = torch.rand(edge_index1.size(1))
    data1 = Data(x=x1, edge_index=edge_index1, edge_weight=edge_weight1)

    x2 = torch.randn(3, in_channels)
    edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight2 = torch.rand(edge_index2.size(1))
    data2 = Data(x=x2, edge_index=edge_index2, edge_weight=edge_weight2)

    conv = StableChebConv(in_channels, out_channels, K=2)

    out1 = conv(x1, edge_index1, edge_weight1)
    out2 = conv(x2, edge_index2, edge_weight2)

    batch = Batch.from_data_list([data1, data2])
    out = conv(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

    assert out.size() == (7, out_channels)
    assert torch.allclose(out1, out[:4], atol=1e-6)
    assert torch.allclose(out2, out[4:], atol=1e-6)


def test_stable_cheb_conv_multi_graph_lambda_max():
    in_channels, out_channels = (16, 16)

    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))
    lambda_max = torch.tensor([2.0, 3.0])

    conv = StableChebConv(in_channels, out_channels, K=3)

    out4 = conv(x, edge_index, edge_weight, batch)
    assert out4.size() == (num_nodes, out_channels)

    out5 = conv(x, edge_index, edge_weight, batch, lambda_max)
    assert out5.size() == (num_nodes, out_channels)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index, edge_weight, batch), out4)
        assert torch.allclose(
            jit(x, edge_index, edge_weight, batch, lambda_max),
            out5,
        )


def test_stable_cheb_conv_projection():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.randn(2, 8)

    conv = StableChebConv(8, 16, K=2)
    assert conv.in_proj is not None

    out = conv(x, edge_index)
    assert out.size() == (2, 16)


def test_stable_cheb_conv_no_projection_when_same_channels():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.randn(2, 16)

    conv = StableChebConv(16, 16, K=2)
    assert conv.in_proj is None

    out = conv(x, edge_index)
    assert out.size() == (2, 16)


def test_stable_cheb_conv_no_bias():
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 8))

    conv = StableChebConv(8, 8, K=3, bias=False)
    assert conv.bias is None

    out = conv(x, edge_index)
    assert out.size() == (num_nodes, 8)


def test_stable_cheb_conv_antisymmetric_weights():
    conv = StableChebConv(8, 8, K=3, gamma=0.1)

    for w in conv.weights:
        A = conv._antisymmetric(w)
        off_diag = A + A.t()
        assert torch.allclose(
            off_diag,
            -2 * conv.gamma * torch.eye(8),
            atol=1e-6,
        ), "A + A^T should equal -2*gamma*I for the antisymm parameterisation"


def test_stable_cheb_conv_antisymmetric_weights_zero_gamma():
    conv = StableChebConv(8, 8, K=3, gamma=0.0)

    for w in conv.weights:
        A = conv._antisymmetric(w)
        assert torch.allclose(A, -A.t(), atol=1e-6), \
            "With gamma=0, A must be skew-symmetric (A = -A^T)"


def test_stable_cheb_conv_residual_connection():
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.randn(2, 8)

    conv = StableChebConv(8, 8, K=1, epsilon=0.5, gamma=0.0, bias=False)

    for w in conv.weights:
        torch.nn.init.zeros_(w)

    out = conv(x, edge_index)
    assert torch.allclose(out, x, atol=1e-6), \
        "With zero weights the output must equal the input (pure residual)"


def test_stable_cheb_conv_epsilon_scaling():
    torch.manual_seed(42)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 8))

    conv_small = StableChebConv(8, 8, K=2, epsilon=0.1)
    conv_large = StableChebConv(8, 8, K=2, epsilon=1.0)

    for ws, wl in zip(conv_small.weights, conv_large.weights):
        wl.data.copy_(ws.data)
    if conv_small.bias is not None:
        conv_large.bias.data.copy_(conv_small.bias.data)

    out_small = conv_small(x, edge_index)
    out_large = conv_large(x, edge_index)

    diff_small = (out_small - x).norm()
    diff_large = (out_large - x).norm()

    assert diff_small < diff_large, ("Smaller epsilon must produce a smaller "
                                     "deviation from the residual input")


def test_stable_cheb_conv_k1():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 8))

    conv = StableChebConv(8, 8, K=1)
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, 8)


def test_stable_cheb_conv_large_k():
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 8))

    conv = StableChebConv(8, 8, K=10)
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, 8)
    assert not torch.isnan(out).any(), "Output must be finite for K=10"
    assert not torch.isinf(out).any(), "Output must be finite for K=10"


def test_stable_cheb_conv_no_normalization():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 8))

    conv = StableChebConv(8, 8, K=3, normalization=None)
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, 8)


def test_stable_cheb_conv_rw_normalization():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 8))

    conv = StableChebConv(8, 8, K=3, normalization='rw')
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, 8)


def test_stable_cheb_conv_invalid_args():
    with pytest.raises(AssertionError):
        StableChebConv(8, 8, K=0)

    with pytest.raises(AssertionError):
        StableChebConv(8, 8, K=2, normalization='invalid')

    with pytest.raises(AssertionError):
        StableChebConv(8, 8, K=2, epsilon=-0.1)

    with pytest.raises(AssertionError):
        StableChebConv(8, 8, K=2, gamma=-1.0)


def test_stable_cheb_conv_weight_shapes():
    in_channels, out_channels, K = 8, 16, 4
    conv = StableChebConv(in_channels, out_channels, K=K)

    assert len(conv.weights) == K
    for w in conv.weights:
        assert w.shape == (out_channels, out_channels), \
            "Weight matrices must be square in out_channels dimension"


def test_stable_cheb_conv_gradient_flow():
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 8), requires_grad=True)

    conv = StableChebConv(8, 8, K=3)
    out = conv(x, edge_index)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "Gradients must be finite"
    for w in conv.weights:
        assert w.grad is not None
        assert not torch.isnan(w.grad).any(), "Weight gradients must be finite"


def test_stable_cheb_conv_repr():
    conv = StableChebConv(16, 32, K=4, epsilon=0.3, gamma=0.05,
                          normalization='rw')
    assert repr(conv) == ("StableChebConv(16, 32, K=4, "
                          "epsilon=0.3, gamma=0.05, "
                          "normalization='rw')")
