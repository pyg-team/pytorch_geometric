from unittest import TestCase

import torch
from numpy.testing import assert_equal

from .mm_diagonal import mm_diagonal


class MmDiagonalTest(TestCase):
    def test_mm_diagonal(self):
        a = torch.FloatTensor([2, 3, 4])
        index = [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        index = torch.LongTensor(index)
        value = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = torch.sparse.FloatTensor(index, value, torch.Size([3, 3]))
        out = mm_diagonal(a, b)

        expected_out = [
            [2, 4, 6],
            [12, 15, 18],
            [28, 32, 36],
        ]

        assert_equal(out.to_dense().numpy(), expected_out)

    def test_mm_diagonal_transpose(self):
        a = torch.FloatTensor([2, 3, 4])
        index = [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        index = torch.LongTensor(index)
        value = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = torch.sparse.FloatTensor(index, value, torch.Size([3, 3]))
        out = mm_diagonal(a, b, transpose=True)

        expected_out = [
            [2, 6, 12],
            [8, 15, 24],
            [14, 24, 36],
        ]

        assert_equal(out.to_dense().numpy(), expected_out)
