import torch

from torch_geometric.nn import DiffGroupNorm
from torch_geometric.testing import is_full_test, withDevice


@withDevice
def test_diff_group_norm(device):
    x = torch.randn(6, 16, device=device)

    norm = DiffGroupNorm(16, groups=4, lamda=0, device=device)
    assert str(norm) == 'DiffGroupNorm(16, groups=4)'

    assert torch.allclose(norm(x), x)

    if is_full_test():
        jit = torch.jit.script(norm)
        assert torch.allclose(jit(x), x)

    norm = DiffGroupNorm(16, groups=4, lamda=0.01, device=device)
    assert str(norm) == 'DiffGroupNorm(16, groups=4)'

    out = norm(x)
    assert out.size() == x.size()

    if is_full_test():
        jit = torch.jit.script(norm)
        assert torch.allclose(jit(x), out)


def test_group_distance_ratio():
    x = torch.randn(6, 16)
    y = torch.tensor([0, 1, 0, 1, 1, 1])

    assert DiffGroupNorm.group_distance_ratio(x, y) > 0

    if is_full_test():
        jit = torch.jit.script(DiffGroupNorm.group_distance_ratio)
        assert jit(x, y) > 0
