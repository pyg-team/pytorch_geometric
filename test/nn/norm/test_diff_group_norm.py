import torch

from torch_geometric.nn import DiffGroupNorm
from torch_geometric.testing import is_full_test


def test_diff_group_norm():
    x = torch.randn(6, 16)

    norm = DiffGroupNorm(16, groups=4, lamda=0)
    assert norm.__repr__() == 'DiffGroupNorm(16, groups=4)'

    assert norm(x).tolist() == x.tolist()

    if is_full_test():
        jit = torch.jit.script(norm)
        assert jit(x).tolist() == x.tolist()

    norm = DiffGroupNorm(16, groups=4, lamda=0.01)
    assert norm.__repr__() == 'DiffGroupNorm(16, groups=4)'

    out = norm(x)
    assert out.size() == x.size()

    if is_full_test():
        jit = torch.jit.script(norm)
        assert jit(x).tolist() == out.tolist()


def test_group_distance_ratio():
    x = torch.randn(6, 16)
    y = torch.tensor([0, 1, 0, 1, 1, 1])

    assert DiffGroupNorm.group_distance_ratio(x, y) > 0

    if is_full_test():
        jit = torch.jit.script(DiffGroupNorm.group_distance_ratio)
        assert jit(x, y) > 0
