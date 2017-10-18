from math import pi as PI

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from torch_geometric.graph.geometry import (vec2ang, polar_coordinates,
                                            edges_from_faces)


class GeometryTest(TestCase):
    def test_vec2ang(self):
        x = torch.FloatTensor([0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0])
        y = torch.FloatTensor([0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1])

        expected = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

        assert_almost_equal((vec2ang(x, y) / PI).numpy(), expected, 2)

    def test_2d_polar_coordinates(self):
        vertices = torch.FloatTensor([[-1, -0], [0, -1], [1, 0], [0, 1]])
        edges = torch.LongTensor([
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ])

        dists, angs = polar_coordinates(vertices, edges)

        expected_dists = [1.4, 2, 1.4, 1.4, 1.4, 2, 1.4, 1.4, 1.4, 1.4]
        assert_almost_equal(dists.numpy(), expected_dists, 1)

        expected_angs = [1.75, 2, 0.25, 0.75, 0.25, 1, 1.25, 0.75, 1.25, 1.75]
        assert_almost_equal((angs / PI).numpy(), expected_angs, 2)

    def test_3d_polar_coordinates(self):
        vertices = torch.FloatTensor([[0, 0, 0], [1, 1, 1]])
        edges = torch.LongTensor([[0, 1], [1, 0]])

        dists, angs1, angs2 = polar_coordinates(vertices, edges)

        expected_dists = [1.73, 1.73]
        assert_almost_equal(dists.numpy(), expected_dists, 2)

        expected_angs1 = [0.25, 1.25]
        assert_almost_equal((angs1 / PI).numpy(), expected_angs1, 2)

        expected_angs2 = [0.25, 1.25]
        assert_almost_equal((angs2 / PI).numpy(), expected_angs2, 2)

    def test_edges_from_faces(self):
        faces = torch.LongTensor([[2, 3, 0], [1, 0, 2]])

        expected = [
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ]

        assert_equal(edges_from_faces(faces).numpy(), expected)
