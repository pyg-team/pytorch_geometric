from unittest import TestCase
import torch
from numpy.testing import assert_equal

from .grid import grid, grid_position


class GridTest(TestCase):
    def test_grid_with_connectivity_4(self):
        adj = grid(torch.Size([3, 2]))

        expected_adj = [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ]

        assert_equal(adj.to_dense().numpy(), expected_adj)

    def test_grid_with_connectivity_8(self):
        adj = grid(torch.Size([3, 2]), connectivity=8)

        expected_adj = [
            [0, 1, 1, 2, 0, 0],
            [1, 0, 2, 1, 0, 0],
            [1, 2, 0, 1, 1, 2],
            [2, 1, 1, 0, 2, 1],
            [0, 0, 1, 2, 0, 1],
            [0, 0, 2, 1, 1, 0],
        ]

        assert_equal(adj.to_dense().numpy(), expected_adj)

    def test_grid_position(self):
        position = grid_position(torch.Size([3, 2]))
        expected_position = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
        assert_equal(position.numpy(), expected_position)
