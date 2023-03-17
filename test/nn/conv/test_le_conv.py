import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import LEConv
from torch_geometric.testing import is_full_test


def test_le_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    row, col = edge_index
    x = torch.randn((num_nodes, in_channels))
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_coo_tensor()

    conv = LEConv(in_channels, out_channels)
    assert str(conv) == 'LEConv(16, 32)'
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        torch.allclose(jit(x, edge_index), out, atol=1e-6)

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out)
