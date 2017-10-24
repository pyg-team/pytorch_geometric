from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .spline import spline, spline_weights


class SplineTest(TestCase):
    def test_splines(self):
        values = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        values = torch.stack([values, values], dim=1)
        kernel_size = torch.LongTensor([5, 4])
        is_open_spline = torch.LongTensor([1, 0])
        amount, index = spline(values, kernel_size, is_open_spline, degree=1)

        # Expected open splines.
        a1 = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i1 = [[1, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 3], [4, 3]]

        # Expected closed splines.
        a2 = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i2 = [[0, 3], [0, 3], [1, 0], [2, 1], [3, 2], [3, 2], [0, 3]]

        assert_almost_equal(amount[:, 0].numpy(), a1, 1)
        assert_almost_equal(amount[:, 1].numpy(), a2, 1)
        assert_equal(index[:, 0].numpy(), i1)
        assert_equal(index[:, 1].numpy(), i2)

    def test_spline_weights(self):
        values = torch.FloatTensor([0.05, 0.25, 0.5, 0.75, 0.95, 1])
        values = torch.stack([values, values], dim=1)
        kernel_size = torch.LongTensor([5, 4])
        is_open_spline = torch.LongTensor([1, 0])
        amount, index = spline(values, kernel_size, is_open_spline, degree=1)
        amount, index = spline_weights(amount, index)

        expected_amount = [
            [0.04, 0.16, 0.16, 0.64],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0.64, 0.16, 0.16, 0.04],
            [0, 0, 0, 1],
        ]

        assert_almost_equal(amount.numpy(), expected_amount, 2)
