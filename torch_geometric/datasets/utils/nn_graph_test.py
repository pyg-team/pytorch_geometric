import unittest
import torch
from numpy.testing import assert_equal

from .nn_graph import nn_graph


class NNGraphTest(unittest.TestCase):
    def test_nn_graph(self):
        pos = [[0, 0], [1, 0], [2, 0], [0, -1], [-2, 0], [0, 2]]
        pos = torch.FloatTensor(pos)
        row, col = nn_graph(pos, k=2)
        expected_row = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        expected_col = [3, 1, 2, 0, 1, 0, 0, 1, 0, 3, 0, 1]

        assert_equal(row.tolist(), expected_row)
        assert_equal(col.tolist(), expected_col)
