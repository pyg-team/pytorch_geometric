import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import RGATConv


def test_rgat_conv():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    edge_attr = torch.randn((4, 8))
    adj = SparseTensor(row=row, col=col, value=edge_attr, sparse_sizes=(4, 4))
    edge_type = torch.tensor([0, 2, 1, 2])

    conv = RGATConv(8, 20, num_relations=4, num_bases=4, mod="additive",
                    attention_mechanism="across-relation",
                    attention_mode="additive-self-attention", heads=2, d=1,
                    edge_dim=8, bias=False)
    assert str(conv) == 'RGATConv(8, 20, heads=2)'
    out = conv(x, edge_index, edge_type, edge_attr)
    assert out.size() == (4, 40)
    assert torch.allclose(
        conv(x, edge_index, edge_type, edge_attr, size=(4, 4)), out)
    assert torch.allclose(conv(x, adj.t(), edge_type), out)

    t = '(Tensor, Tensor, OptTensor, OptTensor, Size, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, edge_index, edge_type),
                          conv(x, edge_index, edge_type))

    conv = RGATConv(8, 20, num_relations=4, num_blocks=2, mod="additive",
                    attention_mechanism="across-relation",
                    attention_mode="additive-self-attention", heads=2, d=1,
                    edge_dim=8, bias=False)
    assert str(conv) == 'RGATConv(8, 20, heads=2)'
    out = conv(x, edge_index, edge_type, edge_attr)
    assert out.size() == (4, 40)
    assert torch.allclose(
        conv(x, edge_index, edge_type, edge_attr, size=(4, 4)), out)
    assert torch.allclose(conv(x, adj.t(), edge_type), out)
