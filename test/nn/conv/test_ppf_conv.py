import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

import torch_geometric.typing
from torch_geometric.nn import PPFConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_ppf_conv():
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)
    pos2 = torch.randn(2, 3)
    n1 = F.normalize(torch.rand(4, 3), dim=-1)
    n2 = F.normalize(torch.rand(2, 3), dim=-1)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    local_nn = Seq(Lin(16 + 4, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PPFConv(local_nn, global_nn)
    assert str(conv) == (
        'PPFConv(local_nn=Sequential(\n'
        '  (0): Linear(in_features=20, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '), global_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')

    out = conv(x1, pos1, n1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, n1, adj1.t()), out, atol=1e-3)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, pos1, n1, adj2.t()), out, atol=1e-3)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, pos1, n1, edge_index), out, atol=1e-3)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, pos1, n1, adj2.t()), out, atol=1e-3)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    out = conv(x1, (pos1, pos2), (n1, n2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, None), (pos1, pos2), (n1, n2), edge_index),
                          out, atol=1e-3)
    assert torch.allclose(conv(x1, (pos1, pos2), (n1, n2), adj1.t()), out,
                          atol=1e-3)
    assert torch.allclose(conv((x1, None), (pos1, pos2), (n1, n2), adj1.t()),
                          out, atol=1e-3)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv(x1, (pos1, pos2), (n1, n2), adj2.t()), out,
                              atol=1e-3)
        assert torch.allclose(
            conv((x1, None), (pos1, pos2), (n1, n2), adj2.t()), out, atol=1e-3)

    if is_full_test():
        assert torch.allclose(
            jit((x1, None), (pos1, pos2), (n1, n2), edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(
                jit((x1, None), (pos1, pos2), (n1, n2), adj2.t()), out,
                atol=1e-3)
