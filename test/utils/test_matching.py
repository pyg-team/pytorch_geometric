import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import maximal_matching


def test_maximal_matching_undirected():
    n, m = 5, 12
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()

    perm1 = torch.arange(m)
    obs1 = maximal_matching(edge_index=index, num_nodes=n, perm=perm1)
    exp1 = torch.tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).bool()

    assert torch.equal(obs1, exp1)

    perm2 = perm1.roll(-1, 0)
    obs2 = maximal_matching(edge_index=index, num_nodes=n, perm=perm2)
    exp2 = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).bool()

    assert torch.equal(obs2, exp2)

    perm3 = torch.arange(m).flip(-1)
    obs3 = maximal_matching(edge_index=index, num_nodes=n, perm=perm3)
    exp3 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).bool()

    assert torch.equal(obs3, exp3)

    if is_full_test():
        jit = torch.jit.script(maximal_matching)
        assert exp1.equal(jit(index, perm=perm1))
        assert exp2.equal(jit(index, perm=perm2))
        assert exp3.equal(jit(index, perm=perm3))


def test_maximal_matching_directed():
    n, m = 5, 9
    index = torch.tensor([[0, 0, 1, 2, 2, 3, 4, 4, 4],
                          [1, 4, 4, 1, 3, 2, 0, 2, 3]]).long()

    perm1 = torch.arange(m)
    obs1 = maximal_matching(edge_index=index, num_nodes=n, perm=perm1)
    exp1 = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0]).bool()

    assert torch.equal(obs1, exp1)

    perm2 = perm1.roll(-1, 0)
    obs2 = maximal_matching(edge_index=index, num_nodes=n, perm=perm2)
    exp2 = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 0]).bool()

    assert torch.equal(obs2, exp2)

    perm3 = torch.arange(m).flip(-1)
    obs3 = maximal_matching(edge_index=index, num_nodes=n, perm=perm3)
    exp3 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 1]).bool()

    assert torch.equal(obs3, exp3)

    if is_full_test():
        jit = torch.jit.script(maximal_matching)
        assert exp1.equal(jit(index, perm=perm1))
        assert exp2.equal(jit(index, perm=perm2))
        assert exp3.equal(jit(index, perm=perm3))
