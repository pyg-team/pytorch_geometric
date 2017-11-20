from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .spline_cpu import spline_cpu


class SplineCPUTest(TestCase):
    def test_open_spline(self):
        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.LongTensor([5])
        is_open_spline = torch.LongTensor([1])

        amount, index = spline_cpu(input, kernel_size, is_open_spline, 1)

        a = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i = [[1, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 3], [0, 4]]

        assert_almost_equal(amount.numpy(), a, 1)
        assert_equal(index.numpy(), i)

    def test_closed_spline(self):
        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.LongTensor([4])
        is_open_spline = torch.LongTensor([0])

        amount, index = spline_cpu(input, kernel_size, is_open_spline, 1)

        a = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i = [[1, 0], [1, 0], [2, 1], [3, 2], [0, 3], [0, 3], [1, 0]]

        assert_almost_equal(amount.numpy(), a, 1)
        assert_equal(index.numpy(), i)

    def test_spline_2d(self):
        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        input = torch.stack([input, input], dim=1)
        kernel_size = torch.LongTensor([5, 4])
        is_open_spline = torch.LongTensor([1, 0])
        amount, index = spline_cpu(input, kernel_size, is_open_spline, 1)

        expected_amount = [
            [0, 0, 0, 1],
            [0.04, 0.16, 0.16, 0.64],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0.64, 0.16, 0.16, 0.04],
            [0, 0, 0, 1],
        ]
        expected_index = [
            [5, 4, 1, 0],
            [5, 4, 1, 0],
            [10, 9, 6, 5],
            [15, 14, 11, 10],
            [16, 19, 12, 15],
            [16, 19, 12, 15],
            [1, 0, 17, 16],
        ]

        assert_almost_equal(amount.numpy(), expected_amount, 2)
        assert_equal(index.numpy(), expected_index)

    def test_spline_3d(self):
        input = torch.FloatTensor([0.05, 0.25, 0.5, 0.75, 0.95, 1])
        input = torch.stack([input, input, input], dim=1)
        kernel_size = torch.LongTensor([5, 4, 4])
        is_open_spline = torch.LongTensor([1, 0, 0])
        amount, index = spline_cpu(input, kernel_size, is_open_spline, 1)

        self.assertEqual(list(amount.size()), [6, 8])
        self.assertEqual(list(index.size()), [6, 8])
        self.assertLess(index.max(), 5 * 4 * 4)
