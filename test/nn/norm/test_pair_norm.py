import torch
from torch_geometric.nn import PairNorm


def test_pair_norm():
    norm = PairNorm(mode="None")
    assert norm.__repr__() == ('PairNorm(mode=None)')
    x = torch.randn(100, 16)
    assert norm(x).size() == (100, 16)

    norm = PairNorm(mode="PN")
    assert norm.__repr__() == ('PairNorm(mode=PN)')
    x = torch.randn(100, 16)
    assert norm(x).size() == (100, 16)

    norm = PairNorm(mode="PN-SI")
    assert norm.__repr__() == ('PairNorm(mode=PN-SI)')
    x = torch.randn(100, 16)
    assert norm(x).size() == (100, 16)

    norm = PairNorm(mode="PN-SCS")
    assert norm.__repr__() == ('PairNorm(mode=PN-SCS)')
    x = torch.randn(100, 16)
    assert norm(x).size() == (100, 16)
