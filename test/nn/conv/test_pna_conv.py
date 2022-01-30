import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import PNAConv

aggregators = ['sum', 'mean', 'min', 'max', 'var', 'std']
scalers = [
    'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'
]


def test_pna_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0), 3)
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = PNAConv(16, 32, aggregators, scalers,
                   deg=torch.tensor([0, 3, 0, 1]), edge_dim=3, towers=4)
    assert conv.__repr__() == 'PNAConv(16, 32, towers=4, edge_dim=3)'
    out = conv(x, edge_index, value)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj.t()), out, atol=1e-6)

    t = '(Tensor, Tensor, OptTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, edge_index, value), out, atol=1e-6)

    t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, adj.t()), out, atol=1e-6)
