import torch
import copy
from torch_geometric.nn.conv.Bayesian_GCNConv import BayesianGCNConv
from torch_geometric.testing import is_full_test


def test_gcn_conv_with_sparse_input_feature():
    x = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [0, 1]]),
                                values=torch.tensor([1., 1.]),
                                size=torch.Size([4, 16]))
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0,
                                                    0]])
    conv = BayesianGCNConv(16, 32)
    out, kl = conv(x, edge_index)
    print(out.shape, kl.detach())
    assert out.size() == (4, 32)


def test_static_gcn_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0,
                                                    0]])
    conv = BayesianGCNConv(16, 32)
    out, kl = conv(x, edge_index)
    print(out.shape, kl.detach().cpu())
    assert out.size() == (3, 4, 32)


def test_gcn_conv_with_decomposed_layers():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0,
                                                    0]])
    conv = BayesianGCNConv(16, 32)
    decomposed_conv = copy.deepcopy(conv)
    decomposed_conv.decomposed_layers = 2
    out1, kl = conv(x, edge_index)
    out2, kl_2 = decomposed_conv(x, edge_index)
    print(out1.shape, out2.shape)
    # the weight differs from out1 to out2 because different weight and biases
    assert torch.allclose(out1, out2, atol=1.0)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(decomposed_conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()


def test_gcn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0,
                                                    0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    # zeros_values = torch.zeros(value.size()).to(device)
    # adj1 = torch.sparse_coo_tensor(edge_index, value, device=x.device)
    # adj2 = torch.sparse_coo_tensor(edge_index, zeros_value, device=x.device)
    conv = BayesianGCNConv(16, 32)
    assert conv.__repr__() == 'BayesianGCNConv(16, 32)'
    out1, kl = conv(x, edge_index)
    print(out1.shape, kl.detach())
    assert out1.size() == (4, 32)

    out2, kl = conv(x, edge_index, value)
    print(out2.shape, kl.detach())
    assert out2.size() == (4, 32)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()
