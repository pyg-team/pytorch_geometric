import torch

import torch_geometric.typing
from torch_geometric.nn import MLP, PointGNNConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_point_gnn_conv():
    x = torch.randn(6, 8)
    pos = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 5], [1, 2, 3, 4, 3, 4]])
    adj1 = to_torch_csc_tensor(edge_index, size=(6, 6))

    conv = PointGNNConv(
        mlp_h=MLP([8, 16, 3]),
        mlp_f=MLP([3 + 8, 16, 8]),
        mlp_g=MLP([8, 16, 8]),
    )
    assert str(conv) == ('PointGNNConv(\n'
                         '  mlp_h=MLP(8, 16, 3),\n'
                         '  mlp_f=MLP(11, 16, 8),\n'
                         '  mlp_g=MLP(8, 16, 8),\n'
                         ')')

    out = conv(x, pos, edge_index)
    assert out.size() == (6, 8)
    assert torch.allclose(conv(x, pos, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert torch.allclose(conv(x, pos, adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, pos, edge_index), out, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, pos, adj2.t()), out, atol=1e-6)
