from math import pi as PI

from unittest import TestCase
import torch
# from torch.autograd import Variable
from numpy.testing import assert_equal

from torch_geometric.nn.functional.spline_gcn import (closed_spline,
                                                      open_spline, spline_gcn)
from torch_geometric.graph.geometry import mesh_adj


class SplineGcnTest(TestCase):
    def test_closed_spline(self):
        values = torch.FloatTensor([0, 0.125, 0.5, 1, 1.5, 1.875, 2]) * PI

        B = [[0, 0.25, 0, 0, 0, 0.75, 0], [1, 0.75, 1, 1, 1, 0.25, 1]]
        C = [[0, 0, 1, 2, 3, 3, 0], [3, 3, 0, 1, 2, 2, 3]]

        out_B, out_C = closed_spline(values, partitions=4)
        assert_equal(out_B.numpy(), B)
        assert_equal(out_C.numpy(), C)

    def test_open_spline(self):
        values = torch.FloatTensor([0, 0.125, 0.5, 1, 1.5, 1.875, 2]) * PI

        B = [[0, 0.25, 0, 0, 0, 0.75, 0], [1, 0.75, 1, 1, 1, 0.25, 1]]
        C = [[1, 1, 2, 3, 4, 4, 4], [0, 0, 1, 2, 3, 3, 3]]

        out_B, out_C = open_spline(values, partitions=4)
        assert_equal(out_B.numpy(), B)
        assert_equal(out_C.numpy(), C)

    def test_forward(self):
        vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
        edges = [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]]
        adj = mesh_adj(torch.FloatTensor(vertices), torch.LongTensor(edges))

        features = torch.FloatTensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

        weight = torch.FloatTensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ])
        weight = weight.view(3, 4, 1, 1)

        kernel_size = [2, 4]
        spline_degree = 1

        print(adj.to_dense()[:, :, 0])
        print(adj.to_dense()[:, :, 1] / PI)

        spline_gcn(adj, features, weight, kernel_size, spline_degree)
