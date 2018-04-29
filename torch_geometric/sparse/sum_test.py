from unittest import TestCase

import torch
from numpy.testing import assert_equal

from .sum import sum


class SumTest(TestCase):
    def test_sum(self):
        index = torch.LongTensor([[0, 0, 1, 3], [1, 2, 0, 2]])
        value = torch.FloatTensor([4, 3, 2, 3])
        a = torch.sparse.FloatTensor(index, value, torch.Size([4, 3]))

        expected_a = [
            [0, 4, 3],
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 3],
        ]

        assert_equal(a.to_dense().numpy(), expected_a)
        assert_equal(sum(a).item(), 12)
        assert_equal(sum(a, dim=0).numpy(), [2, 4, 6])
        assert_equal(sum(a, dim=1).numpy(), [7, 2, 0, 3])
