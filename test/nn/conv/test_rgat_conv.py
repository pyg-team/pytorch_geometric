import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import RGATConv


@pytest.mark.parametrize('mod', [
    'additive',
    'scaled',
    'f-additive',
    'f-scaled',
])
@pytest.mark.parametrize('attention_mechanism', [
    'within-relation',
    'across-relation',
])
@pytest.mark.parametrize('attention_mode', [
    'additive-self-attention',
    'multiplicative-self-attention',
])
def test_rgat_conv(mod, attention_mechanism, attention_mode):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = torch.tensor([0, 2, 1, 2])
    edge_attr = torch.randn((4, 8))

    conv = RGATConv(8, 20, num_relations=4, num_bases=4, mod=mod,
                    attention_mechanism=attention_mechanism,
                    attention_mode=attention_mode, heads=2, dim=1, edge_dim=8)
    assert str(conv) == 'RGATConv(8, 20, heads=2)'

    out = conv(x, edge_index, edge_type, edge_attr)
    assert out.size() == (4, 40)


def test_rgat_conv_jittable():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    edge_attr = torch.randn((4, 8))
    adj = SparseTensor(row=row, col=col, value=edge_attr, sparse_sizes=(4, 4))
    edge_type = torch.tensor([0, 2, 1, 2])

    conv = RGATConv(8, 20, num_relations=4, num_bases=4, mod='additive',
                    attention_mechanism='across-relation',
                    attention_mode='additive-self-attention', heads=2, dim=1,
                    edge_dim=8, bias=False)

    out = conv(x, edge_index, edge_type, edge_attr)
    assert out.size() == (4, 40)
    assert torch.allclose(conv(x, adj.t(), edge_type), out)

    t = '(Tensor, Tensor, OptTensor, OptTensor, Size, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, edge_index, edge_type),
                          conv(x, edge_index, edge_type))
