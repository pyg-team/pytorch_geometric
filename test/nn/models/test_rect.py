import torch

import torch_geometric.typing
from torch_geometric.nn import RECT_L
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_rect():
    x = torch.randn(6, 8)
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = torch.randint(0, 2, (6, ), dtype=torch.bool)

    model = RECT_L(8, 16)
    assert str(model) == 'RECT_L(8, 16)'

    out = model(x, edge_index)
    assert out.size() == (6, 8)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert torch.allclose(out, model(x, adj.t()), atol=1e-6)

    # Test `embed`:
    embed_out = model.embed(x, edge_index)
    assert embed_out.size() == (6, 16)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(embed_out, model.embed(x, adj.t()), atol=1e-6)

    # Test `get_semantic_labels`:
    labeds_out = model.get_semantic_labels(x, y, mask)
    assert labeds_out.size() == (int(mask.sum()), 8)

    if is_full_test():
        jit = torch.jit.script(model.jittable())
        assert torch.allclose(jit(x, edge_index), out, atol=1e-6)
        assert torch.allclose(embed_out, jit.embed(x, edge_index), atol=1e-6)
        assert torch.allclose(labeds_out, jit.get_semantic_labels(x, y, mask))

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        jit = torch.jit.script(model.jittable(use_sparse_tensor=True))
        assert torch.allclose(jit(x, adj.t()), out, atol=1e-6)
        assert torch.allclose(embed_out, jit.embed(x, adj.t()), atol=1e-6)
        assert torch.allclose(labeds_out, jit.get_semantic_labels(x, y, mask))
