from unittest import TestCase
import torch
from numpy.testing import assert_equal

from .target_indegree import TargetIndegreeAdj


class TargetIndegreeTest(TestCase):
    def test_target_indegree_adj(self):
        index = torch.LongTensor([[0, 0, 1, 2], [1, 2, 0, 1]])
        weight = torch.FloatTensor([2, 3, 4, 6])
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([3, 3]))

        transform = TargetIndegreeAdj()
        _, adj, _ = transform((None, adj, None))

        expected_adj = [
            [0, 1, 0.375],
            [0.5, 0, 0],
            [0, 1, 0],
        ]

        assert_equal(adj.to_dense().numpy(), expected_adj)
