import torch

# import torch_geometric.typing
from torch_geometric.nn.conv import BayesianGCNConv
from torch_geometric.testing import is_full_test


def test_bayesian_conv():
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
