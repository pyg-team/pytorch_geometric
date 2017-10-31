from math import sqrt

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .polar import PolarAdj


class PolarTest(TestCase):
    def test_polar_adj_2d(self):
        position = torch.LongTensor([[1, 0], [0, 0], [-2, 2]])
        edge = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        transform = PolarAdj()

        position, adj = transform((position, edge))
        adj = adj.to_dense()

        expected_position = [[1, 0], [0, 0], [-2, 2]]
        expected_adj_rho = [
            [0, 1 / sqrt(8), 0],
            [1 / sqrt(8), 0, 1],
            [0, 1, 0],
        ]
        expected_adj_theta = [
            [0, 0.5, 0],
            [0, 0, 0.375],
            [0, 0.875, 0],
        ]

        assert_equal(adj.size(), [3, 3, 2])
        assert_equal(position.numpy(), expected_position)
        assert_almost_equal(adj[:, :, 0].numpy(), expected_adj_rho)
        assert_almost_equal(adj[:, :, 1].numpy(), expected_adj_theta)

    def test_polar_adj_3d(self):
        position = torch.LongTensor([[1, 0, 0], [1, 0, 1], [-1, 2, 1]])
        edge = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        transform = PolarAdj()

        position, adj = transform((position, edge))
        adj = adj.to_dense()

        expected_position = [[1, 0, 0], [1, 0, 1], [-1, 2, 1]]
        expected_adj_rho = [
            [0, 1 / sqrt(8), 0],
            [1 / sqrt(8), 0, 1],
            [0, 1, 0],
        ]
        expected_adj_theta = [
            [0, 0, 0],
            [0, 0, 0.375],
            [0, 0.875, 0],
        ]
        expected_adj_phi = [
            [0, 0, 0],
            [1, 0, 0.5],
            [0, 0.5, 0],
        ]

        assert_equal(adj.size(), [3, 3, 3])
        assert_equal(position.numpy(), expected_position)
        assert_almost_equal(adj[:, :, 0].numpy(), expected_adj_rho)
        assert_almost_equal(adj[:, :, 1].numpy(), expected_adj_theta)
        assert_almost_equal(adj[:, :, 2].numpy(), expected_adj_phi)
