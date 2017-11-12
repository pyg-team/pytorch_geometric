from unittest import TestCase

import torch
from numpy.testing import assert_equal, assert_almost_equal

if torch.cuda.is_available():
    from .spline_cubic_gpu import spline_cubic_gpu


class SplineQuadraticGPUTest(TestCase):
    def test_open_spline(self):
        if not torch.cuda.is_available():
            return

        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.LongTensor([7])
        is_open_spline = torch.LongTensor([1])

        a1, i1 = spline_cubic_gpu(input.cuda(),
                                  kernel_size.cuda(), is_open_spline.cuda(), 7)

        a2 = [
            [0.1667, 0.6667, 0.1667, 0],
            [0.0853, 0.6307, 0.2827, 0.0013],
            [0.1667, 0.6667, 0.1667, 0],
            [0.1667, 0.6667, 0.1667, 0],
            [0.1667, 0.6667, 0.1667, 0],
            [0.0013, 0.2827, 0.6307, 0.0853],
            [0.1667, 0.6667, 0.1667, 0],
        ]
        i2 = [[0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5],
              [3, 4, 5, 6], [3, 4, 5, 6], [4, 5, 6, 0]]

        assert_almost_equal(a1.cpu().numpy(), a2, 4)
        assert_equal(i1.cpu().numpy(), i2)

    def test_closed_spline(self):
        if not torch.cuda.is_available():
            return

        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.LongTensor([4])
        is_open_spline = torch.LongTensor([0])

        a1, i1 = spline_cubic_gpu(input.cuda(),
                                  kernel_size.cuda(), is_open_spline.cuda(), 4)

        a2 = [[0.5, 0.5, 0], [0.32, 0.66, 0.02], [0.5, 0.5, 0], [0.5, 0.5, 0],
              [0.5, 0.5, 0], [0.02, 0.66, 0.32], [0.5, 0.5, 0]]
        i2 = [[0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1], [3, 0, 1],
              [0, 1, 2]]

        assert_almost_equal(a1.cpu().numpy(), a2, 2)
        assert_equal(i1.cpu().numpy(), i2)
