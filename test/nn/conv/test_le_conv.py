import torch

import torch_geometric.typing
from torch_geometric.nn import LEConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_le_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = LEConv(16, 32)
    assert str(conv) == 'LEConv(16, 32)'
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        torch.allclose(jit(x, edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj2.t()), out)
