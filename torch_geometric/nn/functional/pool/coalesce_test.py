from unittest import TestCase
import torch

from numpy.testing import assert_equal

from .coalesce import remove_self_loops, coalesce


class CoalesceTest(TestCase):
    def test_remove_self_loops(self):
        index = torch.LongTensor([
            [2, 2, 0, 2, 1, 0, 2, 1, 1],
            [0, 1, 0, 2, 0, 2, 0, 0, 1],
        ])
        expected_index = [[2, 2, 1, 0, 2, 1], [0, 1, 0, 2, 0, 0]]

        index = remove_self_loops(index)
        assert_equal(index.tolist(), expected_index)


    def test_coalesce(self):
        index = torch.LongTensor([[2, 2, 1, 0, 2, 1], [0, 1, 0, 2, 0, 0]])
        expected_index = [[0, 1, 2, 2], [2, 0, 0, 1]]

        index = coalesce(index)
        assert_equal(index.tolist(), expected_index)
