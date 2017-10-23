from math import sqrt

from unittest import TestCase
import torch
# # from torch.autograd import Variable
from numpy.testing import assert_almost_equal

from .spline_gcn import edgewise_spline_gcn, spline_gcn
from ...graph.geometry import mesh_adj


class SplineGcnTest(TestCase):
    def test_edgewise_spline_gcn_forward(self):
        vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))
        values = adj._values()
        features = torch.FloatTensor([[3, 4], [5, 6], [7, 8], [9, 10]])
        weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)

        features = edgewise_spline_gcn(
            values,
            features,
            weight,
            kernel_size=[3, 4],
            max_radius=sqrt(16 + 16),
            degree=1)

        expected_features = [
            [(3 * (0.5 + 3.5 + 4.5 + 7.5) + 4 * (1 + 4 + 5 + 8)) * 0.25],
            [(5 * (4.5 + 5.5) + 6 * (5 + 6)) * 0.5],
            [(7 * (5.5 + 6.5 + 9.5 + 10.5) + 8 * (6 + 7 + 10 + 11)) * 0.25],
            [(9 * (10.5 + 11.5) + 10 * (11 + 12)) * 0.5],
        ]

        assert_almost_equal(features.numpy(), expected_features, 1)

    def test_forward(self):
        vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))
        features = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)

        features = spline_gcn(
            adj,
            features,
            weight,
            kernel_size=[3, 4],
            max_radius=sqrt(16 + 16),
            degree=1)

        expected_features = [
            [426 + 0],
            [0],
            [0],
            [0],
            [0],
        ]

        assert_almost_equal(features.numpy(), expected_features, 1)

    def test_backward(self):
        pass
