from unittest import TestCase
import torch
from numpy.testing import assert_equal

from .geometry import edges_from_faces


class GeometryTest(TestCase):
    def test_edges_from_faces(self):
        faces = torch.LongTensor([[2, 3, 0], [1, 0, 2]])

        expected = [
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 3, 0, 2, 0, 1, 3, 0, 2],
        ]

        assert_equal(edges_from_faces(faces).numpy(), expected)
