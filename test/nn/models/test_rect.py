import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import RECT_L
from torch_geometric.testing import is_full_test


def test_rect():
    x = torch.randn(6, 8)
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
    mask = torch.randint(0, 2, (6, ), dtype=torch.bool)

    model = RECT_L(8, 16)
    assert str(model) == 'RECT_L(8, 16)'

    out = model(x, edge_index)
    assert out.size() == (6, 8)
    assert torch.allclose(out, model(x, adj.t()))

    # Test `embed`:
    embed_out = model.embed(x, edge_index)
    assert embed_out.size() == (6, 16)
    assert torch.allclose(embed_out, model.embed(x, adj.t()))

    # Test `get_semantic_labels`:
    labeds_out = model.get_semantic_labels(x, y, mask)
    assert labeds_out.size() == (int(mask.sum()), 8)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(model.jittable(t))
        assert torch.allclose(jit(x, edge_index), out)
        assert torch.allclose(embed_out, jit.embed(x, edge_index))
        assert torch.allclose(labeds_out, jit.get_semantic_labels(x, y, mask))

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(model.jittable(t))
        assert torch.allclose(jit(x, adj.t()), out)
        assert torch.allclose(embed_out, jit.embed(x, adj.t()))
        assert torch.allclose(labeds_out, jit.get_semantic_labels(x, y, mask))
