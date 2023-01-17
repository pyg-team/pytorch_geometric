import torch

from torch_geometric.utils import maximal_independent_set


def test_independent_set():
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()
    rank = torch.arange(5)

    jit = torch.jit.script(maximal_independent_set)

    mis = jit(index, k=0, rank=rank)
    assert mis.equal(torch.ones_like(mis))

    mis = jit(index, k=1, rank=rank)
    assert mis.equal(torch.tensor([1, 0, 1, 0, 0]).bool())

    mis = jit(index, k=2, rank=rank)
    assert mis.equal(torch.tensor([1, 0, 0, 0, 0]).bool())
