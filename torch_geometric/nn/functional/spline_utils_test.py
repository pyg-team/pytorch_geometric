from math import pi as PI

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .spline_utils import open_spline, closed_spline


class SplineUtilsTest(TestCase):
    def test_open_spline(self):
        values = torch.FloatTensor([0, 0.1, 0.5, 1, 1.5, 1.9, 2]) * PI

        B = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        C = [[1, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 3], [4, 3]]

        out_B, out_C = open_spline(values, num_knots=4)
        assert_almost_equal(out_B.numpy(), B, 1)
        assert_equal(out_C.numpy(), C)

    def test_closed_spline(self):
        values = torch.FloatTensor([0, 0.1, 0.5, 1, 1.5, 1.9, 2]) * PI

        B = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        C = [[0, 3], [0, 3], [1, 0], [2, 1], [3, 2], [3, 2], [0, 3]]

        out_B, out_C = closed_spline(values, num_knots=4)
        assert_almost_equal(out_B.numpy(), B, 1)
        assert_equal(out_C.numpy(), C)
