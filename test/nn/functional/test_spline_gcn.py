from math import pi as PI

from unittest import TestCase
import torch
# from torch.autograd import Variable
from numpy.testing import assert_equal, assert_almost_equal

from torch_geometric.nn.functional.spline_gcn import (angle_spline,
                                                      radius_spline)


class SplineGcnTest(TestCase):
    def test_forward(self):
        pass
        # i = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        # v1 = torch.FloatTensor([1, 2, 3, 4, 5, 6])
        # v2 = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
        #                         [11, 12]])
        # a = torch.sparse.FloatTensor(i, v2, torch.Size([3, 3, 2]))

    def test_angle_spline(self):
        values = torch.FloatTensor([
            0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
            0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1
        ]) * 2 * PI
        B, C = angle_spline(values, partitions=4)

        assert_almost_equal(B.numpy(), [
            [0.25, 0.75],
            [0.5, 0.5],
            [0.75, 0.25],
            [0, 1],
            [0.25, 0.75],
            [0.5, 0.5],
            [0.75, 0.25],
            [0, 1],
            [0.25, 0.75],
            [0.5, 0.5],
            [0.75, 0.25],
            [0, 1],
            [0.25, 0.75],
            [0.5, 0.5],
            [0.75, 0.25],
            [0, 1],
        ], 6)
        assert_equal(C.numpy(), [
            [0, 3],
            [0, 3],
            [0, 3],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 2],
            [3, 2],
            [3, 2],
            [3, 2],
            [0, 3],
        ])

    def test_radius_spline(self):
        values = torch.FloatTensor([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
        B, C = radius_spline(values, partitions=2)

        assert_equal(B.numpy(), [
            [1, 0],
            [0.75, 0.25],
            [0.5, 0.5],
            [0.25, 0.75],
            [1, 0],
            [0.75, 0.25],
            [0.5, 0.5],
            [0.25, 0.75],
            [1, 0],
        ])
        assert_equal(
            C.numpy(),
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [2, 1],  # A bit strange, but there's no corresponding value.
            ])
