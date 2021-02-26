import torch
from torch_geometric.nn import DiffNorm


def test_diff_norm():
    h = torch.tensor([[0, 0], [1, 1], [2, 2], [3.0, 3.0], [4, 4], [3.1, 1]])

    dn = DiffNorm(lamda=0)
    assert torch.all(dn(h) == h)

    torch.jit.script(dn)

    dn = DiffNorm(lamda=0.01)
    assert dn(h).shape == h.shape


def test_groupdist():
    h = torch.tensor([[0, 0], [1, 1], [2, 2], [3.0, 3.0], [4, 4], [3.1, 1]])
    c = torch.tensor([1, 2, 1, 2, 2, 2], dtype=int).reshape(-1, 1)

    assert DiffNorm.groupDistanceRatio(h, c) > 0
