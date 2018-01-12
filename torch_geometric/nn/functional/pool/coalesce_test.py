from unittest import TestCase
import torch

from numpy.testing import assert_equal

from .coalesce import coalesce


class CoalesceTest(TestCase):
    def test_coalesce(self):
        index = torch.LongTensor([[2, 2, 1, 0, 2, 1], [0, 1, 0, 2, 0, 0]])
        expected_index = [[0, 1, 2, 2], [2, 0, 0, 1]]

        index = coalesce(index)
        assert_equal(index.tolist(), expected_index)
