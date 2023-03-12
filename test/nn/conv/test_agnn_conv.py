import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import AGNNConv
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('requires_grad', [True, False])
def test_agnn_conv(requires_grad):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_coo_tensor()

    conv = AGNNConv(requires_grad=requires_grad)
    assert str(conv) == 'AGNNConv()'
    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out.tolist()

        t = '(Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out, atol=1e-6)
