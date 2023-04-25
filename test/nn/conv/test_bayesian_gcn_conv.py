import copy

import torch

import torch_geometric.typing
from torch_geometric.nn.conv import BayesianGCNConv
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_bayesian_conv_norm():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))
    conv = BayesianGCNConv(16, 32)
    assert conv.__repr__() == "BayesianGCNConv(16, 32)"
    out1, kl_out1 = conv(x, edge_index)
    out1_adj, kl1_adj = conv(x, adj1.t())
    assert out1.size() == (4, 32)

    assert torch.allclose(out1_adj.mean(), out1.mean(), atol=1.0)
    assert torch.allclose(torch.var(out1), torch.var(out1_adj), atol=1)
    assert torch.allclose(kl_out1, kl1_adj, atol=1e-6)

    out2, kl_out2 = conv(x, edge_index, value)
    out2_adj, kl2_adj = conv(x, adj2.t())
    assert out2.size() == (4, 32)

    assert torch.allclose(out2_adj.mean(), out2.mean(), atol=1.0)
    assert torch.allclose(torch.var(out2), torch.var(out2_adj), atol=1)
    assert torch.allclose(kl_out2, kl2_adj, atol=1e-6)

    # Implementation with sparse tensor uncomment it to test on another support
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        out3, kl3 = conv(x, adj3.t())
        out4, kl4 = conv(x, adj4.t())
        assert torch.allclose(out3.mean(), out1.mean(), atol=1.0)
        assert torch.allclose(out4.mean(), out2.mean(), atol=1.0)
        assert torch.allclose(torch.var(out1), torch.var(out3), atol=1)
        assert torch.allclose(torch.var(out4), torch.var(out2), atol=1)
        assert torch.allclose(kl3, kl_out1, atol=1e-6)
        assert torch.allclose(kl4, kl_out2, atol=1e-6)

    # For testing; check it for later
    # conv.cached = True
    # out1_prime, kl_prime = conv(x, edge_index)
    # out1_prime2, kl_prime2 = conv(x, adj1.t())
    # assert conv._cached_edge_index is not None
    # assert torch.allclose(out1_prime.mean(), out1.mean(), atol=1.0)
    # assert torch.allclose(out1_prime2.mean(), out1.mean(), atol=1.0)

    # if torch_geometric.typing.WITH_TORCH_SPARSE:
    #    out1_prime3, kl_prime3 = conv(x, adj3.t())
    #    assert conv._cached_adj_t is not None
    #    assert torch.allclose(out1_prime3.mean(), out1.mean(), atol=1.0)


def test_static_bayesian_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    conv = BayesianGCNConv(16, 32)
    out, kl = conv(x, edge_index)
    assert out.size() == (3, 4, 32)


def test_gcn_conv_with_sparse_input_feature():
    x = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.tensor([1.0, 1.0]),
        size=torch.Size([4, 16]),
    )
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    conv = BayesianGCNConv(16, 32)
    out, kl = conv(x, edge_index)
    assert out.size() == (4, 32)


def test_gcn_conv_with_decomposed_layers():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    conv = BayesianGCNConv(16, 32)
    decomposed_conv = copy.deepcopy(conv)
    decomposed_conv.decomposed_layers = 2
    out1, kl_out1 = conv(x, edge_index)
    out2, kl_out2 = decomposed_conv(x, edge_index)
    # Reminder when declared, the default values for bayesian conv is
    # prior_mean=0, prior_variance=1
    # posterior_mu_init = 0
    # posterior_rho_init = -3.0
    # The KL divergence should be the same for both convs
    assert torch.allclose(torch.abs(out1.mean()), torch.abs(out2.mean()),
                          atol=1)
    assert torch.allclose(torch.var(out1), torch.var(out2), atol=1)
    assert torch.allclose(kl_out1, kl_out2, atol=1e-6)

    if is_full_test():
        t = "(Tensor, Tensor, OptTensor) -> Tensor"
        jit = torch.jit.script(decomposed_conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()


def test_gcn_conv_norm():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
    row, col = edge_index

    conv = BayesianGCNConv(16, 32, flow="source_to_target")
    out1, kl1 = conv(x, edge_index)
    conv.flow = "target_to_source"
    out2, kl2 = conv(x, edge_index.flip(0))
    assert torch.allclose(out1, out2, atol=1.0)
    assert torch.allclose(kl1, kl2, atol=1e-6)


def test_gcn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))

    adj1 = torch.randint(0, 4, (2, 6))
    adj2 = torch.randint(0, 4, (2, 6))

    conv = BayesianGCNConv(16, 32)
    assert conv.__repr__() == "BayesianGCNConv(16, 32)"
    out1, kl_out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    out_1_adj1, kl_adj1 = conv(x, adj1)
    assert torch.allclose(torch.abs(out1.mean()), torch.abs(out_1_adj1.mean()),
                          atol=1)
    assert torch.allclose(torch.var(out1), torch.var(out_1_adj1), atol=1)
    assert torch.allclose(kl_out1, kl_adj1, atol=1e-6)
    out2, kl_out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    out_2_adj2, kl_adj2 = conv(x, adj2)
    assert torch.allclose(torch.abs(out2.mean()), torch.abs(out_2_adj2.mean()),
                          atol=1)
    assert torch.allclose(torch.var(out2), torch.var(out_2_adj2), atol=1)
    assert torch.allclose(kl_out2, kl_adj2, atol=1e-6)

    if is_full_test():
        t = "(Tensor, Tensor, OptTensor) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()

        t = "(Tensor, Tensor, OptTensor) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out1, atol=1e-6)
        assert torch.allclose(jit(x, adj2.t()), out2, atol=1e-6)

    conv.cached = True
    out_f1, kl_f1 = conv(x, edge_index)
    for i in range(len(out1.tolist())):
        assert torch.allclose(torch.abs(out_f1[i].mean()),
                              torch.abs(out1[i].mean()), atol=1)
        assert torch.allclose(torch.var(out_f1[i]), torch.var(out1[i]), atol=1)
    assert torch.allclose(kl_f1, kl_out1, atol=1e-6)
