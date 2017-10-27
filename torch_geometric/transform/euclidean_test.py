from unittest import TestCase
import torch
from numpy.testing import assert_equal

from .euclidean import EuclideanAdj


class EuclideanTest(TestCase):
    def test_euclidean_adj(self):
        position = torch.LongTensor([[1, 0], [0, 0], [-2, 4]])
        edge = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        transform = EuclideanAdj()

        position, adj = transform((position, edge))
        adj = adj.to_dense()

        expected_position = [[1, 0], [0, 0], [-2, 4]]
        expected_adj_x = [
            [0., 0.375, 0],
            [0.625, 0, 0.25],
            [0, 0.75, 0],
        ]
        expected_adj_y = [
            [0., 0.5, 0],
            [0.5, 0, 1],
            [0, 0, 0],
        ]

        assert_equal(adj.size(), [3, 3, 2])
        assert_equal(position.numpy(), expected_position)
        assert_equal(adj[:, :, 0].numpy(), expected_adj_x)
        assert_equal(adj[:, :, 1].numpy(), expected_adj_y)
