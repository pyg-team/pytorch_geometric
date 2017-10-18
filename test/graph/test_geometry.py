from math import pi as PI

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from torch_geometric.graph.geometry import (vec2ang, polar_coordinates,
                                            mesh_adj, edges_from_faces)


class GeometryTest(TestCase):
    def test_vec2ang(self):
        x = torch.FloatTensor([0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1])
        y = torch.FloatTensor([0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0])

        expected = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

        assert_almost_equal((vec2ang(x, y) / PI).numpy(), expected, 2)

    def test_2d_polar_coordinates(self):
        vertices = torch.FloatTensor([[-1, -0], [0, -1], [1, 0], [0, 1]])
        edges = torch.LongTensor([
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ])

        rho, theta = polar_coordinates(vertices, edges).t()

        expected_rho = [1.4, 2, 1.4, 1.4, 1.4, 2, 1.4, 1.4, 1.4, 1.4]
        assert_almost_equal(rho.numpy(), expected_rho, 1)

        expected_theta = [1.75, 2, 0.25, 0.75, 0.25, 1, 1.25, 0.75, 1.25, 1.75]
        assert_almost_equal((theta / PI).numpy(), expected_theta, 2)

    def test_3d_polar_coordinates(self):
        vertices = torch.FloatTensor([[0, 0, 0], [1, 1, 1]])
        edges = torch.LongTensor([[0, 1], [1, 0]])

        rho, theta, phi = polar_coordinates(vertices, edges).t()

        expected_rho = [1.73, 1.73]
        assert_almost_equal(rho.numpy(), expected_rho, 2)

        expected_theta = [0.25, 1.25]
        assert_almost_equal((theta / PI).numpy(), expected_theta, 2)

        expected_phi = [0.25, 1.25]
        assert_almost_equal((phi / PI).numpy(), expected_phi, 2)

    def test_mesh_adj(self):
        vertices = torch.FloatTensor([[1, 0], [0, 0], [-1, 0]])
        edges = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        adj = mesh_adj(vertices, edges).to_dense()

        expected_rho = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_equal(adj[:, :, 0].numpy(), expected_rho)

        expected_theta = [[0, 1, 0], [2, 0, 1], [0, 2, 0]]
        assert_almost_equal((adj[:, :, 1] / PI).numpy(), expected_theta, 1)

    def test_edges_from_faces(self):
        faces = torch.LongTensor([[2, 3, 0], [1, 0, 2]])

        expected = [
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ]

        assert_equal(edges_from_faces(faces).numpy(), expected)
