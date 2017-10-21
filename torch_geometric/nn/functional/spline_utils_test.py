from math import pi as PI, sqrt

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .spline_utils import (open_spline_amount, open_spline_index,
                           closed_spline_amount, closed_spline_index, rescale,
                           create_mask, weight_amount, weight_index,
                           spline_weights)
from ...graph.geometry import mesh_adj


class SplineUtilsTest(TestCase):
    def test_open_spline(self):
        values = torch.FloatTensor([0, 0.2, 1, 2, 3, 3.8, 4])

        a = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i = [[1, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 3], [4, 3]]

        amount = open_spline_amount(values, degree=1)
        index = open_spline_index(values, kernel_size=4, degree=1)

        assert_almost_equal(amount.numpy(), a, 1)
        assert_equal(index.numpy(), i)

    def test_closed_spline(self):
        values = torch.FloatTensor([0, 0.2, 1, 2, 3, 3.8, 4])

        a = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i = [[0, 3], [0, 3], [1, 0], [2, 1], [3, 2], [3, 2], [0, 3]]

        amount = closed_spline_amount(values, degree=1)
        index = closed_spline_index(values, kernel_size=4, degree=1)

        assert_almost_equal(amount.numpy(), a, 1)
        assert_equal(index.numpy(), i)

    def test_rescale(self):
        values = [[1, 0.5 * PI], [2, PI], [3, 1.5 * PI], [5, 2 * PI]]

        values = rescale(
            torch.FloatTensor(values), kernel_size=[3, 4], max_radius=4)

        expected_values = [[0.5, 1], [1, 2], [1.5, 3], [2, 4]]

        assert_equal(values.numpy(), expected_values)

    def test_create_mask(self):
        mask = create_mask(dim=3, degree=1)
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

    def test_weight_amount(self):
        pass

    def test_weight_index(self):

        pass

    def test_spline_weights(self):
        vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))

        amount, index = spline_weights(
            adj._values(), kernel_size=[3, 4], max_radius=sqrt(32), degree=1)

        expected_amount = [
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0, 0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0, 0],
        ]
        expected_index = [
            [4, 7, 0, 3],
            [5, 4, 1, 0],
            [10, 9, 6, 5],
            [11, 10, 7, 6],
        ]

        assert_almost_equal(amount.numpy(), expected_amount, 2)
        assert_equal(index.numpy(), expected_index)
