import pytest
import torch

from torch_geometric.nn import SAGEConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('project', [False, True])
@pytest.mark.parametrize('aggr', ['mean', 'sum'])
def test_sage_conv(project, aggr):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_csc_tensor()

    conv = SAGEConv(8, 32, project=project, aggr=aggr)
    assert str(conv) == f'SAGEConv(8, 32, aggr={aggr})'
    out = conv(x1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, size=(4, 4)), out, atol=1e-6)
    assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out, atol=1e-6)
        assert torch.allclose(jit(x1, edge_index, size=(4, 4)), out, atol=1e-6)

        t = '(Tensor, SparseTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj1.t()), out, atol=1e-6)

    adj1 = adj1.sparse_resize((4, 2))
    adj2 = adj1.to_torch_sparse_csc_tensor()
    conv = SAGEConv((8, 16), 32, project=project, aggr=aggr)
    assert str(conv) == f'SAGEConv((8, 16), 32, aggr={aggr})'
    out1 = conv((x1, x2), edge_index)
    out2 = conv((x1, None), edge_index, (4, 2))
    assert out1.size() == (2, 32)
    assert out2.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, (4, 2)), out1, atol=1e-6)
    assert torch.allclose(conv((x1, x2), adj1.t()), out1, atol=1e-6)
    assert torch.allclose(conv((x1, None), adj1.t()), out2, atol=1e-6)
    assert torch.allclose(conv((x1, x2), adj2.t()), out1, atol=1e-6)
    assert torch.allclose(conv((x1, None), adj2.t()), out2, atol=1e-6)

    if is_full_test():
        t = '(OptPairTensor, Tensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out1, atol=1e-6)
        assert torch.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1,
                              atol=1e-6)
        assert torch.allclose(jit((x1, None), edge_index, size=(4, 2)), out2,
                              atol=1e-6)

        t = '(OptPairTensor, SparseTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj1.t()), out1, atol=1e-6)
        assert torch.allclose(jit((x1, None), adj1.t()), out2, atol=1e-6)


def test_lstm_aggr_sage_conv():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = SAGEConv(8, 32, aggr='lstm')
    assert str(conv) == 'SAGEConv(8, 32, aggr=lstm)'
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj.t()), out)

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 0]])
    with pytest.raises(ValueError, match="'index' tensor is not sorted"):
        conv(x, edge_index)


@pytest.mark.parametrize('aggr_kwargs', [
    dict(mode='cat'),
    dict(mode='proj', mode_kwargs=dict(in_channels=8, out_channels=16)),
    dict(mode='attn', mode_kwargs=dict(in_channels=8, out_channels=16,
                                       num_heads=4)),
    dict(mode='sum'),
])
def test_multi_aggr_sage_conv(aggr_kwargs):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    aggr_kwargs['aggrs_kwargs'] = [{}, {}, {}, dict(learn=True, t=1)]
    conv = SAGEConv(8, 32, aggr=['mean', 'max', 'sum', 'softmax'],
                    aggr_kwargs=aggr_kwargs)
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj.t()), out)
