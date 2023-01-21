import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import maximal_independent_set


def test_independent_set():
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()
    perm = torch.arange(5)

    mis1 = maximal_independent_set(index, k=0, perm=perm)
    assert mis1.equal(torch.ones_like(mis1))

    mis2 = maximal_independent_set(index, k=1, perm=perm)
    assert mis2.equal(torch.tensor([1, 0, 1, 0, 0]).bool())

    mis3 = maximal_independent_set(index, k=2, perm=perm)
    assert mis3.equal(torch.tensor([1, 0, 0, 0, 0]).bool())

    if is_full_test():
        jit = torch.jit.script(maximal_independent_set)
        assert mis1.equal(jit(index, k=0, perm=perm))
        assert mis2.equal(jit(index, k=1, perm=perm))
        assert mis3.equal(jit(index, k=2, perm=perm))
