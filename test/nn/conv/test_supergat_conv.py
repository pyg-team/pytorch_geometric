import torch
from torch_geometric.nn import SuperGATConv
import pytest


@pytest.mark.parametrize('attention_type', ['MX', 'SD'])
def test_supergat_conv(attention_type):

    x_1 = torch.randn(4, 8)
    edge_index_1 = torch.tensor([[0, 1, 2, 3],
                                 [0, 0, 1, 1]])

    conv = SuperGATConv(8, 32, heads=2, attention_type=attention_type,
                        neg_sample_ratio=1.0, edge_sample_ratio=1.0)
    repr_ans = 'SuperGATConv(8, 32, heads=2, type={})'.format(attention_type)
    assert conv.__repr__() == repr_ans

    out_1 = conv(x_1, edge_index_1)
    assert out_1.size() == (4, 64)

    # Negative samples are given.
    neg_edge_index = conv.negative_sampling(edge_index_1, x_1.size(0))
    assert conv(x_1, edge_index_1, neg_edge_index).tolist() == out_1.tolist()
    att_loss_1 = SuperGATConv.get_attention_loss(conv)
    assert isinstance(att_loss_1, torch.Tensor) and att_loss_1 > 0

    # Jit: No negative samples are given.
    t = '(Tensor, Tensor, NoneType, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x_1, edge_index_1).tolist() == out_1.tolist()
    att_loss_jit = SuperGATConv.get_attention_loss(jit)
    assert isinstance(att_loss_jit, torch.Tensor) and att_loss_jit > 0

    # Jit: Negative samples are given.
    t = '(Tensor, Tensor, Tensor, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x_1, edge_index_1, neg_edge_index).tolist() == out_1.tolist()
    assert torch.allclose(SuperGATConv.get_attention_loss(jit),
                          att_loss_1, atol=1e-6)

    x_2 = torch.randn(8, 8)
    edge_index_2 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                 [0, 0, 1, 1, 4, 4, 5, 5]])
    batch_2 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out_2 = conv(x_2, edge_index_2, batch=batch_2)
    assert out_2.size() == (8, 64)

    # Batch & negative samples are given.
    neg_edge_index_2 = conv.negative_sampling(edge_index_2, x_2.size(0),
                                              batch_2)
    assert conv(x_2, edge_index_2, neg_edge_index_2).tolist() == out_2.tolist()
    att_loss_2 = SuperGATConv.get_attention_loss(conv)
    assert isinstance(att_loss_2, torch.Tensor) and att_loss_2 > 0

    # Jit: Batch & negative samples are given.
    t = '(Tensor, Tensor, Tensor, Tensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    out_2_jit = jit(x_2, edge_index_2, neg_edge_index_2, batch_2)
    assert out_2_jit.tolist() == out_2.tolist()
    assert torch.allclose(SuperGATConv.get_attention_loss(jit),
                          att_loss_2, atol=1e-6)
