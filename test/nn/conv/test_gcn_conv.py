import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.testing import is_full_test
from torch_geometric.typing import WITH_PT21, SparseTensor
from torch_geometric.utils import to_torch_coo_tensor, to_torch_csc_tensor


def test_gcn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = GCNConv(16, 32)
    assert str(conv) == 'GCNConv(16, 32)'

    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
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

    conv.cached = True
    conv(x, edge_index)
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, edge_index), out1, atol=1e-6)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        conv(x, adj3.t())
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)


def test_gcn_conv_with_decomposed_layers():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    def hook(module, inputs):
        assert inputs[0]['x_j'].size() == (10, 32 // module.decomposed_layers)

    conv = GCNConv(16, 32)
    conv.register_message_forward_pre_hook(hook)
    out1 = conv(x, edge_index)

    conv.decomposed_layers = 2
    assert conv.propagate.__module__.endswith('message_passing')
    out2 = conv(x, edge_index)
    assert torch.allclose(out1, out2)

    # TorchScript should still work since it relies on class methods
    # (but without decomposition).
    torch.jit.script(conv)

    conv.decomposed_layers = 1
    assert conv.propagate.__module__.endswith('GCNConv_propagate')


def test_gcn_conv_with_sparse_input_feature():
    x = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.tensor([1., 1.]),
        size=torch.Size([4, 16]),
    )
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GCNConv(16, 32)
    assert conv(x, edge_index).size() == (4, 32)


def test_static_gcn_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GCNConv(16, 32)
    out = conv(x, edge_index)
    assert out.size() == (3, 4, 32)


def test_gcn_conv_error():
    with pytest.raises(ValueError, match="does not support adding self-loops"):
        GCNConv(16, 32, normalize=False, add_self_loops=True)


def test_gcn_conv_flow():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])

    conv = GCNConv(16, 32, flow="source_to_target")
    out1 = conv(x, edge_index)
    conv.flow = "target_to_source"
    out2 = conv(x, edge_index.flip(0))
    assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('layout', [torch.sparse_coo, torch.sparse_csr])
def test_gcn_norm_gradient(requires_grad, layout):
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = torch.ones(edge_index.size(1), requires_grad=requires_grad)
    adj = to_torch_coo_tensor(edge_index, edge_weight)
    if layout == torch.sparse_csr:
        adj = adj.to_sparse_csr()

    # TODO Sparse CSR tensor doesn't inherit `requires_grad` for PyTorch < 2.1.
    if layout == torch.sparse_csr and not WITH_PT21:
        assert not gcn_norm(adj)[0].requires_grad
    else:
        assert adj.requires_grad == gcn_norm(adj)[0].requires_grad
