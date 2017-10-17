from math import pi as PI

from unittest import TestCase
import torch
from numpy.testing import assert_equal

from torch_geometric.graph.geometry import (vec2ang, polar_coordinates,
                                            edges_from_faces)


class GeometryTest(TestCase):
    def test_vec2ang(self):
        y = torch.FloatTensor([0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0])
        x = torch.FloatTensor([0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1])

        expected = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

        assert_equal((vec2ang(y, x) / PI).numpy(), expected)

    def test_polar_coordinates(self):
        pass

    def test_edges_from_faces(self):
        faces = torch.LongTensor([[2, 3, 0], [1, 0, 2]])

        expected = [
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ]

        assert_equal(edges_from_faces(faces).numpy(), expected)
