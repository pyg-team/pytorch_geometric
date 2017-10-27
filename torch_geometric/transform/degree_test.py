from __future__ import division

from math import sqrt

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .degree import DegreeAdj


class DegreeTest(TestCase):
    def test_degree_adj(self):
        weight = torch.FloatTensor([2, 3, 4, 6])
        edge = torch.LongTensor([[0, 0, 1, 2], [1, 2, 0, 1]])
        adj = torch.sparse.FloatTensor(edge, weight, torch.Size([3, 3]))

        transform = DegreeAdj()

        _, adj = transform((None, adj))
        adj = adj.to_dense()

        expected_adj_out = [
            [0, 1 / sqrt(5), 1 / sqrt(5)],
            [1 / sqrt(4), 0, 0],
            [0, 1 / sqrt(6), 0],
        ]
        expected_adj_in = [
            [0, 1 / sqrt(4), 1 / sqrt(6)],
            [1 / sqrt(5), 0, 0],
            [0, 1 / sqrt(4), 0],
        ]

        assert_equal(adj.size(), [3, 3, 2])
        assert_almost_equal(adj[:, :, 0].numpy(), expected_adj_out, 8)
        assert_almost_equal(adj[:, :, 1].numpy(), expected_adj_in, 8)
