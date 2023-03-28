import copy

import torch

from torch_geometric.nn.conv.BayesianGCNConv import BayesianGCNConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_bayesian_conv_norm():
    x = torch.randn(4, 16)
    x_0 = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)
    conv = BayesianGCNConv(16, 32)
    assert conv.__repr__() == 'BayesianGCNConv(16, 32)'
    out1, kl_out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    out_1_adj1, kl_adj1 = conv(x, adj1)
    assert torch.allclose(out1.mean(), out_1_adj1.mean(), atol=0.1)
    assert torch.allclose(torch.var(out1), torch.var(out_1_adj1), atol=1)
    assert torch.allclose(kl_out1, kl_adj1, atol=1e-6)

    out2, kl2 = conv(x, edge_index, value)
    out2_adj2, kl_adj2 =  conv(x, adj2)
    assert torch.allclose(out2.mean(), out2_adj2.mean(), atol=0.1)
    assert torch.allclose(torch.var(out2), torch.var(out2_adj2), atol=1)
    assert torch.allclose(kl2, kl_adj2, atol=1e-6)



def test_static_bayesian_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    conv = BayesianGCNConv(16, 32)
    out, kl = conv(x, edge_index)
    assert out.size() == (3, 4, 32)


def test_gcn_conv_with_sparse_input_feature():
    x = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [0, 1]]),
                                values=torch.tensor([1., 1.]),
                                size=torch.Size([4, 16]))
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
    # The outputs of the Bayesian conv layer should have a mean close to 0
    # The variance should not exceed 1
    # The KL divergence should be the same for both convs
    assert torch.allclose(out1.mean(), out2.mean(), atol=0.1)
    assert torch.allclose(torch.var(out1), torch.var(out2), atol=1)
    assert torch.allclose(kl_out1, kl_out2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(decomposed_conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()


def test_gcn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))

    adj1 = torch.randint(0, 4, (2, 6))
    adj2 = torch.randint(0, 4, (2, 6))

    conv = BayesianGCNConv(16, 32)
    assert conv.__repr__() == 'BayesianGCNConv(16, 32)'
    out1, kl_out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    out_1_adj1, kl_adj1 = conv(x, adj1)
    assert torch.allclose(out1.mean(), out_1_adj1.mean(), atol=0.1)
    assert torch.allclose(torch.var(out1), torch.var(out_1_adj1), atol=1)
    assert torch.allclose(kl_out1, kl_adj1, atol=1e-6)
    out2, kl_out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    out_2_adj2, kl_adj2 = conv(x, adj2)
    assert torch.allclose(out2.mean(), out_2_adj2.mean(), atol=0.1)
    assert torch.allclose(torch.var(out2), torch.var(out_2_adj2), atol=1)
    assert torch.allclose(kl_out2, kl_adj2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()

        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out1, atol=1e-6)
        assert torch.allclose(jit(x, adj2.t()), out2, atol=1e-6)

    conv.cached = True
    out_f1, kl_f1 = conv(x, edge_index)
    for i in range(len(out1.tolist())):
        assert torch.allclose(out_f1[i].mean(), out1[i].mean(), atol=0.1)
        assert torch.allclose(torch.var(out_f1[i]), torch.var(out1[i]), atol=1)
    assert torch.allclose(kl_f1, kl_out1, atol=1e-6)
