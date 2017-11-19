from unittest import TestCase
import torch
from numpy.testing import assert_equal

from ..datasets.data import Data
from .cartesian import CartesianAdj


class CartesianTest(TestCase):
    def test_cartesian_adj(self):
        index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        weight = torch.FloatTensor([1, 1, 1, 1])
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([3, 3]))
        position = torch.FloatTensor([[1, 0], [0, 0], [-2, 4]])
        data = Data(None, adj, position, None)

        data = CartesianAdj()(data)
        adj, position = data.adj, data.position

        expected_adj = [
            [[0, 0], [0.375, 0.5], [0, 0]],
            [[0.625, 0.5], [0, 0], [0.25, 1]],
            [[0, 0], [0.75, 0], [0, 0]],
        ]
        expected_position = [[1, 0], [0, 0], [-2, 4]]

        assert_equal(adj.to_dense().numpy(), expected_adj)
        assert_equal(position.numpy(), expected_position)
