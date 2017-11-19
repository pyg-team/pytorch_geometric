from math import sqrt

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from ..datasets.data import Data
from .polar import PolarAdj


class PolarTest(TestCase):
    def test_polar_adj(self):
        index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        weight = torch.FloatTensor([1, 1, 1, 1])
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([3, 3]))
        position = torch.FloatTensor([[1, 0], [0, 0], [-2, 2]])
        data = Data(None, adj, position, None)

        data = PolarAdj()(data)
        adj, position = data.adj, data.position

        expected_adj = [
            [[0, 0], [1 / sqrt(8), 0.5], [0, 0]],
            [[1 / sqrt(8), 0], [0, 0], [1, 0.375]],
            [[0, 0], [1, 0.875], [0, 0]],
        ]
        expected_position = [[1, 0], [0, 0], [-2, 2]]

        assert_almost_equal(adj.to_dense().numpy(), expected_adj)
        assert_equal(position.numpy(), expected_position)
