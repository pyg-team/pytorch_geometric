from math import pi as PI

from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from torch_geometric.graph.geometry import (vec2ang, polar_coordinates,
                                            edges_from_faces)


class GeometryTest(TestCase):
    def test_vec2ang(self):
        y = torch.FloatTensor([0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0])
        x = torch.FloatTensor([0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1])

        expected = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

        assert_equal((vec2ang(y, x) / PI).numpy(), expected)

    def test_2d_polar_coordinates(self):
        vertices = torch.FloatTensor([[0, -1], [-1, 0], [0, 1], [1, 0]])
        edges = torch.LongTensor([
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ])

        dists, angs = polar_coordinates(vertices, edges)

        expected_dists = [1.4, 2, 1.4, 1.4, 1.4, 2, 1.4, 1.4, 1.4, 1.4]
        assert_almost_equal(dists.numpy(), expected_dists, 1)

        expected_angs = [1.75, 2, 0.25, 0.75, 0.25, 1, 1.25, 0.75, 1.25, 1.75]
        assert_equal((angs / PI).numpy(), expected_angs)

    def test_edges_from_faces(self):
        faces = torch.LongTensor([[2, 3, 0], [1, 0, 2]])

        expected = [
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ]

        assert_equal(edges_from_faces(faces).numpy(), expected)
