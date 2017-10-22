from math import sqrt

from unittest import TestCase
import torch
# # from torch.autograd import Variable
from numpy.testing import assert_equal, assert_almost_equal

from .spline_gcn import edgewise_spline_gcn, spline_gcn
from ...graph.geometry import mesh_adj


class SplineGcnTest(TestCase):
    def test_edgewise_spline_gcn(self):
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

    #         edgewise_spline_gcn(val)

    # def edgewise_spline_gcn(values,
    #                         features,
    #                         weight,
    #                         kernel_size,
    #                         max_radius,
    #                         degree=1):

    #     def test_closed_spline(self):
    #         values = torch.FloatTensor([0, 0.125, 0.5, 1, 1.5, 1.875, 2]) * PI

    #         B = [[0, 0.25, 0, 0, 0, 0.75, 0], [1, 0.75, 1, 1, 1, 0.25, 1]]
    #         C = [[0, 0, 1, 2, 3, 3, 0], [3, 3, 0, 1, 2, 2, 3]]

    #         out_B, out_C = closed_spline(values, partitions=4)
    #         assert_equal(out_B.numpy(), B)
    #         assert_equal(out_C.numpy(), C)

    #     def test_open_spline(self):
    #         values = torch.FloatTensor([0, 0.125, 0.5, 1, 1.5, 1.875, 2]) * PI

    #         B = [[0, 0.25, 0, 0, 0, 0.75, 0], [1, 0.75, 1, 1, 1, 0.25, 1]]
    #         C = [[1, 1, 2, 3, 4, 4, 4], [0, 0, 1, 2, 3, 3, 3]]

    #         out_B, out_C = open_spline(values, partitions=4)
    #         assert_equal(out_B.numpy(), B)
    #         assert_equal(out_C.numpy(), C)

    #     def test_points(self):
    #         p = list(points(3, 2))
    #       assert_equal(p, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
    #                          [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    #     def test_transform(self):
    #         vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    #         edges = [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]]
    #         adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))

    #        features = torch.FloatTensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    #         weight = torch.FloatTensor([
    #             [0.1, 0.2, 0.3, 0.4],
    #             [0.5, 0.6, 0.7, 0.8],
    #             [0.9, 1.0, 1.1, 1.2],
    #             [1.3, 1.4, 1.5, 1.6],
    #             [1.7, 1.8, 1.9, 2.0],
    #             [2.1, 2.2, 2.3, 2.4],
    #         ])
    #         weight = weight.view(2, 1, 3, 4)

    #         transform(adj, features, weight, spline_degree=1)

    #     def test_weight_indices(self):
    #         # spline_indices should have shape [|E|, dim, degree + 1].
    #         # Test dim = 2, spline_degree = 1
    #       # We test two points with coordinates (1, 1) and (3, -3) with radius 4.
    #         spline_indices = [[1, 0], [0, 3]]
    #         spline_indices = torch.LongTensor(spline_indices)
    #         kernel_size = [3, 4]
    #         weight_indices(spline_indices, kernel_size)
    #         # 0, 3, 4, 7

    #         spline_indices = [[2, 1], [3, 2]]
    #         spline_indices = torch.LongTensor(spline_indices)
    #         kernel_size = [3, 4]
    #         weight_indices(spline_indices, kernel_size)
    #         # 6, 7, 10, 11

    #         spline_indices = [[3, 4], [0, 1], [2, 3]]
    #         spline_indices = torch.LongTensor(spline_indices)
    #         kernel_size = [5, 3, 4]
    #         weight_indices(spline_indices, kernel_size)
    #         # 38, 39, 42, 43, 50, 51, 54, 55

    #     def test_weight_amounts(self):
    #         # spline_indices should have shape [|E|, dim, degree + 1].
    #         # Test dim = 2, spline_degree = 1
    #       # We test two points with coordinates (1, 1) and (3, -3) with radius 4.
    #         spline_amounts = [[0.25, 0.75], [0.4, 0.6]]
    #         spline_amounts = torch.FloatTensor(spline_amounts)
    #         kernel_size = [3, 4]
    #         weight_amounts(spline_amounts, kernel_size)
    #         # 0, 3, 4, 7

    #     #     pass

    #     # def test_weight_amounts(self):
    #     #     pass

    #     # def test_forward(self):
    #     #     return
    #     #     vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    #     #     edges = [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]]
    #    #     adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))

    #        features = torch.FloatTensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    #     #     weight = torch.FloatTensor([
    #     #         [0.1, 0.2, 0.3, 0.4],
    #     #         [0.5, 0.6, 0.7, 0.8],
    #     #         [0.9, 1.0, 1.1, 1.2],
    #     #     ])
    #     #     weight = weight.view(1, 1, 3 * 4)

    #     #     kernel_size = [2, 4]
    #     #     spline_degree = 1

    #     #     print(adj.to_dense()[:, :, 0])
    #     #     print(adj.to_dense()[:, :, 1] / PI)

    #     #     spline_gcn(adj, features, weight, kernel_size, spline_degree)
