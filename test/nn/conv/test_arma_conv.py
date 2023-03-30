import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import ARMAConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_arma_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = ARMAConv(16, 32, num_stacks=8, num_layers=4)
    assert str(conv) == 'ARMAConv(16, 32, num_stacks=8, num_layers=4)'
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    with pytest.raises(RuntimeError):  # No 3D feature tensor support.
        assert torch.allclose(conv(x, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj2.t()), out, atol=1e-6)


def test_lazy_arma_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = ARMAConv(-1, 32, num_stacks=8, num_layers=4)
    assert str(conv) == 'ARMAConv(-1, 32, num_stacks=8, num_layers=4)'
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
