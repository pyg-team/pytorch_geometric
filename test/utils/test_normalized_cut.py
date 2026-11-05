import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import normalized_cut


def test_normalized_cut():
    row = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    col = torch.tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3])
    edge_attr = torch.tensor(
        [3.0, 3.0, 6.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 2.0])
    expected = torch.tensor([4.0, 4.0, 5.0, 2.5, 5.0, 1.0, 2.5, 2.0, 1.0, 2.0])

    out = normalized_cut(torch.stack([row, col], dim=0), edge_attr)
    assert torch.allclose(out, expected)

    if is_full_test():
        jit = torch.jit.script(normalized_cut)
        out = jit(torch.stack([row, col], dim=0), edge_attr)
        assert torch.allclose(out, expected)


def test_normalized_cut_multidimensional_edge_attr():
    # The degree scale must align with the edge axis (dim 0), not broadcast
    # over the feature axis.
    row = torch.tensor([1, 1, 2, 3])
    col = torch.tensor([3, 3, 1, 2])
    edge_index = torch.stack([row, col], dim=0)

    # Destination degrees at nodes 1, 2, 3 are 1, 1, 2, so edge scales are
    # [1.5, 1.5, 2.0, 1.5].
    scale = torch.tensor([1.5, 1.5, 2.0, 1.5])

    # (E, F) with F != E must not raise.
    edge_attr = torch.ones(4, 2)
    out = normalized_cut(edge_index, edge_attr)
    assert out.shape == (4, 2)
    assert torch.allclose(out, scale.view(4, 1) * edge_attr)

    # (E, E) must scale rows, not columns.
    edge_attr = torch.arange(1., 17.).view(4, 4)
    out = normalized_cut(edge_index, edge_attr)
    assert torch.allclose(out, edge_attr * scale.view(4, 1))

    # Higher-rank features (E, F, G).
    edge_attr = torch.ones(4, 2, 3)
    out = normalized_cut(edge_index, edge_attr)
    assert out.shape == (4, 2, 3)
    assert torch.allclose(out, edge_attr * scale.view(4, 1, 1))

    # Empty (0, F) features.
    edge_attr = torch.empty(0, 2)
    out = normalized_cut(torch.empty(2, 0, dtype=torch.long), edge_attr)
    assert out.shape == (0, 2)
