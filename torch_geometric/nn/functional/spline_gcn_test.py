from math import sqrt

from unittest import TestCase
import torch
from torch.autograd import Variable, gradcheck
from numpy.testing import assert_almost_equal

from .spline_gcn import edgewise_spline_gcn, _EdgewiseSplineGcn, spline_gcn
from ...graph.geometry import mesh_adj


class SplineGcnTest(TestCase):
    def test_edgewise_spline_gcn_forward(self):
        return
        vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))
        values = adj._values()
        features = torch.FloatTensor([[3, 4], [5, 6], [7, 8], [9, 10]])
        features = Variable(features)
        weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)
        weight = Variable(weight)

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

        assert_almost_equal(features.data.numpy(), expected_features, 1)

    def test_edgewise_spline_gcn_backward(self):
        return
        vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        vertices = torch.LongTensor(vertices)
        edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        edges = torch.LongTensor(edges)
        adj = mesh_adj(vertices, edges, type=torch.DoubleTensor)
        values = adj._values()
        features = torch.randn(4, 2).double()
        features = Variable(features, requires_grad=True)
        weight = torch.randn(12, 2, 4).double()
        weight = Variable(weight, requires_grad=True)

        op = _EdgewiseSplineGcn(
            values, kernel_size=[3, 4], max_radius=sqrt(16 + 16), degree=1)

        test = gradcheck(op, (features, weight), eps=1e-6, atol=1e-4)
        self.assertTrue(test)

    def test_forward(self):
        return
        vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))
        features = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        features = Variable(features)
        weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)
        weight = Variable(weight)

        features = spline_gcn(
            adj,
            features,
            weight,
            kernel_size=[3, 4],
            max_radius=sqrt(16 + 16),
            degree=1)

        edgewise_spline_feature = 426

        expected_features = [
            [2 * 1 + 2.5 * 2 + edgewise_spline_feature],
            [2 * 3 + 2.5 * 4],
            [2 * 5 + 2.5 * 6],
            [2 * 7 + 2.5 * 8],
            [2 * 9 + 2.5 * 10],
        ]

        assert_almost_equal(features.data.numpy(), expected_features, 1)

    def test_backward(self):
        return
        vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        vertices = torch.LongTensor(vertices)
        edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        edges = torch.LongTensor(edges)
        adj = mesh_adj(vertices, edges, type=torch.DoubleTensor)
        features = torch.randn(5, 2).double()
        features = Variable(features, requires_grad=True)
        weight = torch.randn(12, 2, 4).double()
        weight = Variable(weight, requires_grad=True)

        def op(features, weight):
            return spline_gcn(
                adj,
                features,
                weight,
                kernel_size=[3, 4],
                max_radius=sqrt(16 + 16),
                degree=1)

        test = gradcheck(op, (features, weight), eps=1e-6, atol=1e-4)
        self.assertTrue(test)
