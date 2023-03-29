import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import RGATConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor


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
    edge_attr = torch.randn((edge_index.size(1), 8))
    edge_type = torch.tensor([0, 2, 1, 2])
    adj1 = to_torch_coo_tensor(edge_index, edge_attr, size=(4, 4))

    conv = RGATConv(8, 20, num_relations=4, num_bases=4, mod='additive',
                    attention_mechanism='across-relation',
                    attention_mode='additive-self-attention', heads=2, dim=1,
                    edge_dim=8, bias=False)

    out = conv(x, edge_index, edge_type, edge_attr)
    assert out.size() == (4, 40)
    # t() expects a tensor with <= 2 sparse and 0 dense dimensions
    adj1_t = adj1.transpose(0, 1).coalesce()
    assert torch.allclose(conv(x, adj1_t, edge_type), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert torch.allclose(conv(x, adj2.t(), edge_type), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, OptTensor, Size, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, edge_index, edge_type),
                              conv(x, edge_index, edge_type))
