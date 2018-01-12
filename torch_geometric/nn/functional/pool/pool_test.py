from unittest import TestCase
import torch

from numpy.testing import assert_equal

from .pool import max_pool, average_pool


class PoolTest(TestCase):
    def test_forward(self):
        input = torch.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        index = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.LongTensor([0, 1, 1, 0])

        expected_max_input = [[5, 3], [8, 7]]
        expected_average_input = [[3, 2.5], [6, 5]]
        expected_index = [[0, 1], [1, 0]]
        expected_position = [[-2, 0], [2, 0]]

        input, index, position = max_pool(input, index, position, cluster)

        input, index, position = max_pool(input, index, position, cluster)
