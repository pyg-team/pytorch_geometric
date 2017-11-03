from __future__ import division

from unittest import TestCase
import torch

from numpy.testing import assert_equal

from .batch_average import batch_average


class BatchAverageTest(TestCase):
    def test_batch_average(self):
        input = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        slice = torch.LongTensor([3, 5])

        output = batch_average(input, slice)

        expected_output = [
            [(1 + 3 + 5) / 3, (2 + 4 + 6) / 3],
            [(7 + 9) / 2, (8 + 10) / 2],
        ]

        assert_equal(output.numpy(), expected_output)
