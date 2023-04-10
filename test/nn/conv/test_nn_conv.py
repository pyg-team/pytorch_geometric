import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

import torch_geometric.typing
from torch_geometric.nn import NNConv
from torch_geometric.testing import is_full_test, withCUDA
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor


@withCUDA
def test_nn_conv(device):
    x1 = torch.randn(4, 8, device=device)
    x2 = torch.randn(2, 16, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device=device)
    value = torch.rand(edge_index.size(1), 3, device=device)
    adj1 = to_torch_coo_tensor(edge_index, value, size=(4, 4))

    nn = Seq(Lin(3, 32), ReLU(), Lin(32, 8 * 32))
    conv = NNConv(8, 32, nn=nn).to(device)
    assert str(conv) == (
        'NNConv(8, 32, aggr=add, nn=Sequential(\n'
        '  (0): Linear(in_features=3, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=256, bias=True)\n'
        '))')

    out = conv(x1, edge_index, value)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, value, size=(4, 4)), out)
    assert torch.allclose(conv(x1, adj1.transpose(0, 1).coalesce()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index, value), out)
        assert torch.allclose(jit(x1, edge_index, value, size=(4, 4)), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj2.t()), out)

    # Test bipartite message passing:
    adj1 = to_torch_coo_tensor(edge_index, value, size=(4, 2))

    conv = NNConv((8, 16), 32, nn=nn).to(device)
    assert str(conv) == (
        'NNConv((8, 16), 32, aggr=add, nn=Sequential(\n'
        '  (0): Linear(in_features=3, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=256, bias=True)\n'
        '))')

    out1 = conv((x1, x2), edge_index, value)
    assert out1.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)
    assert torch.allclose(conv((x1, x2),
                               adj1.transpose(0, 1).coalesce()), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out2.size() == (2, 32)
    assert torch.allclose(conv((x1, None),
                               adj1.transpose(0, 1).coalesce()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert torch.allclose(conv((x1, x2), adj2.t()), out1)
        assert torch.allclose(conv((x1, None), adj2.t()), out2)

    if is_full_test():
        t = '(OptPairTensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index, value), out1)
        assert torch.allclose(jit((x1, x2), edge_index, value, size=(4, 2)),
                              out1)
        assert torch.allclose(jit((x1, None), edge_index, value, size=(4, 2)),
                              out2)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(OptPairTensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj2.t()), out1)
        assert torch.allclose(jit((x1, None), adj2.t()), out2)
