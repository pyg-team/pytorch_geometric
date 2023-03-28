import pytest
import torch

from torch_geometric.nn import SimpleConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('aggr, combine_root', [
    ('mean', None),
    ('sum', 'sum'),
    (['mean', 'max'], 'cat'),
])
def test_simple_conv(aggr, combine_root):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = SimpleConv(aggr, combine_root)
    assert str(conv) == 'SimpleConv()'

    num_aggrs = 1 if isinstance(aggr, str) else len(aggr)
    output_size = sum([8] * num_aggrs) + (8 if combine_root == 'cat' else 0)

    out = conv(x1, edge_index)
    assert out.size() == (4, output_size)
    assert torch.allclose(conv(x1, edge_index, size=(4, 4)), out)
    assert torch.allclose(conv(x1, adj.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out)
        assert torch.allclose(jit(x1, edge_index, size=(4, 4)), out)

        t = '(Tensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj.t()), out)

    adj = adj.sparse_resize((4, 2))

    out = conv((x1, x2), edge_index)
    assert out.size() == (2, output_size)
    assert torch.allclose(conv((x1, x2), edge_index, size=(4, 2)), out)
    assert torch.allclose(conv((x1, x2), adj.t()), out)
