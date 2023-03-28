import torch

from torch_geometric.nn import LEConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_le_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    adj1 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_csc_tensor()

    conv = LEConv(in_channels, out_channels)
    assert str(conv) == 'LEConv(16, 32)'
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)
    assert torch.allclose(conv(x, adj1.t()), out)
    assert torch.allclose(conv(x, adj2.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        torch.allclose(jit(x, edge_index), out)

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out)
