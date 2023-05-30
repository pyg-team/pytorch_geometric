import copy

import pytest
import torch

# import torch_geometric.typing
from torch_geometric.nn import DirGCNConv
from torch_geometric.nn.conv.dir_gcn_conv import dir_gcn_norm
from torch_geometric.testing import is_full_test
# from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor


def test_gcn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    # adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = DirGCNConv(16, 32)
    assert str(conv) == 'DirGCNConv(16, 32)'

    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    # assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    # if torch_geometric.typing.WITH_TORCH_SPARSE:
    #     adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
    #     assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out1, atol=1e-6)

    # if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
    #     t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
    #     jit = torch.jit.script(conv.jittable(t))
    #     assert torch.allclose(jit(x, adj3.t()), out1, atol=1e-6)

    conv.cached = True
    conv(x, edge_index)
    assert conv._cached_edge_index is not None
    assert conv._cached_edge_index_t is not None
    assert torch.allclose(conv(x, edge_index), out1, atol=1e-6)
    # assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    # if torch_geometric.typing.WITH_TORCH_SPARSE:
    #     conv(x, adj3.t())
    #     assert conv._cached_adj_t is not None
    #     assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)


def test_gcn_conv_with_decomposed_layers():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = DirGCNConv(16, 32)

    decomposed_conv = copy.deepcopy(conv)
    decomposed_conv.decomposed_layers = 2

    out1 = conv(x, edge_index)
    out2 = decomposed_conv(x, edge_index)
    assert torch.allclose(out1, out2)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(decomposed_conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out1)


def test_gcn_conv_with_sparse_input_feature():
    x = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.tensor([1., 1.]),
        size=torch.Size([4, 16]),
    )
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = DirGCNConv(16, 32)
    assert conv(x, edge_index).size() == (4, 32)


# def test_static_gcn_conv():
#     x = torch.randn(3, 4, 16)
#     edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

#     conv = DirGCNConv(16, 32)
#     out = conv(x, edge_index)
#     assert out.size() == (3, 4, 32)

# def test_gcn_conv_norm():
#     x = torch.randn(4, 16)
#     edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
#     adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
# adj_t = SparseTensor.from_edge_index(edge_index.flip(0), sparse_sizes=(4, 4))

#     conv = DirGCNConv(16, 32)
#     adj_norm = dir_gcn_norm(adj, num_nodes=4)
#     adj_t_norm = dir_gcn_norm(adj_t, num_nodes=4)
# assert torch.allclose(adj_norm, adj_t_norm.t(), atol=1e-6)


@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('layout', [torch.sparse_coo, torch.sparse_csr])
def test_gcn_norm_gradient(requires_grad, layout):
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = torch.ones(edge_index.size(1), requires_grad=requires_grad)
    adj = to_torch_coo_tensor(edge_index, edge_weight)
    if layout == torch.sparse_csr:
        adj = adj.to_sparse_csr()

    # TODO Sparse CSR tensor does not yet inherit `requires_grad` from `value`.
    if layout == torch.sparse_csr:
        assert not dir_gcn_norm(adj)[0].requires_grad
    else:
        assert adj.requires_grad == dir_gcn_norm(adj)[0].requires_grad
