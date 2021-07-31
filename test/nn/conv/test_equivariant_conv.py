import torch
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn import EquivariantConv


def test_equivariant_conv():
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)
    vel = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    pos_nn = Linear(32, 1)
    vel_nn = Linear(16, 1)
    local_nn = Linear(2 * 16 + 1, 32)
    global_nn = Linear(16 + 32, 16)

    conv = EquivariantConv(local_nn, pos_nn, vel_nn, global_nn)
    assert conv.__repr__() == (
        'EquivariantConv('
        'local_nn=Linear(in_features=33, out_features=32, bias=True),'
        ' pos_nn=Linear(in_features=32, out_features=1, bias=True),'
        ' vel_nn=Linear(in_features=16, out_features=1, bias=True),'
        ' global_nn=Linear(in_features=48, out_features=16, bias=True))')
    out = conv(x1, pos1, edge_index, vel)
    assert out[0].size() == (4, 16)
    assert out[1][0].size() == (4, 3)
    assert out[1][1].size() == (4, 3)
    assert torch.allclose(conv(x1, pos1, edge_index)[0], out[0], atol=1e-6)
    assert conv(x1, pos1, edge_index)[1][1] is None
    assert torch.allclose(conv(x1, pos1, adj.t(), vel)[0], out[0], atol=1e-6)
    assert torch.allclose(conv(x1, pos1, adj.t())[0], out[0], atol=1e-6)

    t = ('(Tensor, Tensor, Tensor, OptTensor, NoneType)'
         ' -> Tuple[Tensor, Tuple[Tensor, OptTensor]]')
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x1, pos1, edge_index, vel)[0], out[0], atol=1e-6)
    assert torch.allclose(jit(x1, pos1, edge_index, vel)[1][0], out[1][0],
                          atol=1e-6)
    assert torch.allclose(jit(x1, pos1, edge_index, vel)[1][1], out[1][1],
                          atol=1e-6)

    t = ('(Tensor, Tensor, Tensor, NoneType, NoneType)'
         ' -> Tuple[Tensor, Tuple[Tensor, OptTensor]]')
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x1, pos1, edge_index)[0], out[0], atol=1e-6)
    assert jit(x1, pos1, edge_index)[1][1] is None

    t = ('(Tensor, Tensor, SparseTensor, OptTensor, NoneType)'
         ' -> Tuple[Tensor, Tuple[Tensor, OptTensor]]')
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x1, pos1, adj.t(), vel)[0], out[0], atol=1e-6)
    assert torch.allclose(jit(x1, pos1, adj.t(), vel)[1][0], out[1][0],
                          atol=1e-6)
    assert torch.allclose(jit(x1, pos1, adj.t(), vel)[1][1], out[1][1],
                          atol=1e-6)
