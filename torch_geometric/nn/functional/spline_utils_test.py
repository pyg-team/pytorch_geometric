from math import pi as PI, sqrt

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .spline_utils import (open_spline_amount, open_spline_index,
                           closed_spline_amount, closed_spline_index,
                           _create_mask, weight_amount, weight_index,
                           spline_weights, splines)
from ...graph.geometry import mesh_adj


class SplineUtilsTest(TestCase):
    def test_splines(self):
        values = torch.FloatTensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
        values = torch.stack([values, values], dim=1)
        # print(values.size())
        kernel_size = torch.LongTensor([5, 4])
        is_open_spline = torch.LongTensor([1, 0])
        amount, index = splines(values, kernel_size, is_open_spline, degree=1)

        # print(amount.size())

        # Expected open splines.
        a1 = [[0, 1], [0.4, 0.6], [0, 1], [0, 1], [0, 1], [0.6, 0.4], [0, 1]]
        i1 = [[1, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 3], [4, 3]]

        # Expected closed splines.
        a2 = [[0, 1], [0.4, 0.6], [0, 1], [0, 1], [0, 1], [0.6, 0.4], [0, 1]]
        i2 = [[0, 3], [0, 3], [1, 0], [2, 1], [3, 2], [3, 2], [0, 3]]

        # print(index[:, 0])
        # print(i1)
        assert_almost_equal(amount[:, 0].numpy(), a1, 1)
        assert_almost_equal(amount[:, 1].numpy(), a2, 1)
        # assert_equal(index[:, 0].numpy(), i1)

        pass

    # def test_open_spline(self):
    #     values = torch.FloatTensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    #     amount = open_spline_amount(values, kernel_size=5, degree=1)
    #     index = open_spline_index(values, kernel_size=5, degree=1)

    #     a = [[0, 1], [0.4, 0.6], [0, 1], [0, 1], [0, 1], [0.6, 0.4], [0, 1]]
    #     i = [[1, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 3], [4, 3]]

    #     assert_almost_equal(amount.numpy(), a, 1)
    #     assert_equal(index.numpy(), i)

    # def test_closed_spline(self):
    #     values = torch.FloatTensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    #     amount = closed_spline_amount(values, kernel_size=4, degree=1)
    #     index = closed_spline_index(values, kernel_size=4, degree=1)

    #     a = [[0, 1], [0.4, 0.6], [0, 1], [0, 1], [0, 1], [0.6, 0.4], [0, 1]]
    #     i = [[0, 3], [0, 3], [1, 0], [2, 1], [3, 2], [3, 2], [0, 3]]

    #     assert_almost_equal(amount.numpy(), a, 1)
    #     assert_equal(index.numpy(), i)

    # def test_create_mask(self):
    #     return
    #     mask = _create_mask(dim=3, degree=1)

    #     expected_mask = [
    #         [0, 2, 4],
    #         [0, 2, 5],
    #         [0, 3, 4],
    #         [0, 3, 5],
    #         [1, 2, 4],
    #         [1, 2, 5],
    #         [1, 3, 4],
    #         [1, 3, 5],
    #     ]

    #     assert_equal(mask.numpy(), expected_mask)

    # def test_weight_amount(self):
    #     values = torch.FloatTensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    #     kernel_size = 5
    #     is_open_spline = True

    #     output = weight_amount(values, kernel_size, is_open_spline, degree=1)

    #     values = torch.FloatTensor([
    #         [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    #         [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    #     ]).t()

    #     kernel_size = [5, 4]
    #     is_open_spline = [True, False]

    #     output = weight_amount(values, kernel_size, is_open_spline, degree=1)
    #     return

    #     values = [[0.2, 0.2], [1, 1], [2, 2], [3, 3], [3.8, 3.8], [4, 4]]
    #     values = torch.FloatTensor(values)

    #     amount = weight_amount(values, degree=1)

    #     expected_amount = [
    #         [0.04, 0.16, 0.16, 0.64],
    #         [0, 0, 0, 1],
    #         [0, 0, 0, 1],
    #         [0, 0, 0, 1],
    #         [0.64, 0.16, 0.16, 0.04],
    #         [0, 0, 0, 1],
    #     ]

    #     assert_almost_equal(amount.numpy(), expected_amount, 2)

    # def test_weight_index(self):
    #     return
    #     values = [[0.2, 0.2], [1, 1], [2, 2], [3, 3], [3.8, 3.8], [4, 4]]
    #     values = torch.FloatTensor(values)
    #     kernel_size = [5, 4]

    #     index = weight_index(values, kernel_size, degree=1)

    #     expected_index = [
    #         [4, 7, 0, 3],
    #         [9, 8, 5, 4],
    #         [14, 13, 10, 9],
    #         [19, 18, 15, 14],
    #         [19, 18, 15, 14],
    #         [16, 19, 12, 15],
    #     ]

    #     assert_equal(index.numpy(), expected_index)

    # def test_spline_weights_2d(self):
    #     return
    #     vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
    #     edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
    #     adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))

    #     amount, index = spline_weights(
    #         adj._values(),
    #         kernel_size=[3, 4],
    #         max_radius=sqrt(16 + 16),
    #         degree=1)

    #     expected_amount = [
    #         [0.25, 0.25, 0.25, 0.25],
    #         [0.5, 0.5, 0, 0],
    #         [0.25, 0.25, 0.25, 0.25],
    #         [0.5, 0.5, 0, 0],
    #     ]
    #     expected_index = [
    #         [4, 7, 0, 3],
    #         [5, 4, 1, 0],
    #         [10, 9, 6, 5],
    #         [11, 10, 7, 6],
    #     ]

    #     assert_almost_equal(amount.numpy(), expected_amount, 2)
    #     assert_equal(index.numpy(), expected_index)

    # def test_spline_weights_3d(self):
    #     return
    #     vertices = [[0, 0, 0], [1, 1, 1]]
    #     edges = [[0], [1]]
    #     adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))

    #     amount, index = spline_weights(
    #         adj._values(),
    #         kernel_size=[3, 4, 4],
    #         max_radius=sqrt(16 + 16 + 16),
    #         degree=1)

    #     expected_amount = [
    #         [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
    #     ]
    #     expected_index = [
    #         [16, 19, 28, 31, 0, 3, 12, 15],
    #     ]

    #     assert_almost_equal(amount.numpy(), expected_amount, 3)
    #     assert_equal(index.numpy(), expected_index)
