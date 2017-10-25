from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .spline import spline, create_mask, spline_weights


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

    def test_create_mask(self):
        mask = create_mask(dim=3, m=2)

        expected_mask = [
            [0, 2, 4],
            [0, 2, 5],
            [0, 3, 4],
            [0, 3, 5],
            [1, 2, 4],
            [1, 2, 5],
            [1, 3, 4],
            [1, 3, 5],
        ]

        assert_equal(mask.numpy(), expected_mask)

    def test_spline_weights(self):
        values = torch.FloatTensor([0.05, 0.25, 0.5, 0.75, 0.95, 1])
        values = torch.stack([values, values], dim=1)
        kernel_size = torch.LongTensor([5, 4])
        is_open_spline = torch.LongTensor([1, 0])
        amount, index = spline_weights(values, kernel_size, is_open_spline, 1)

        expected_amount = [
            [0.04, 0.16, 0.16, 0.64],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0.64, 0.16, 0.16, 0.04],
            [0, 0, 0, 1],
        ]
        expected_index = [
            [4, 7, 0, 3],
            [9, 8, 5, 4],
            [14, 13, 10, 9],
            [19, 18, 15, 14],
            [19, 18, 15, 14],
            [16, 19, 12, 15],
        ]

        assert_almost_equal(amount.numpy(), expected_amount, 2)
        assert_equal(index.numpy(), expected_index)

        values = torch.FloatTensor([0.05, 0.25, 0.5, 0.75, 0.95, 1])
        values = torch.stack([values, values, values], dim=1)
        kernel_size = torch.LongTensor([5, 4, 4])
        is_open_spline = torch.LongTensor([1, 0, 0])
        amount, index = spline_weights(values, kernel_size, is_open_spline, 1)

        self.assertEqual(list(amount.size()), [6, 8])
        self.assertEqual(list(index.size()), [6, 8])
        self.assertLess(index.max(), 5 * 4 * 4)
