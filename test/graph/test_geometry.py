from unittest import TestCase

import torch
from numpy.testing import assert_equal

from torch_geometric.graph.geometry import edges_from_faces


class GeometryTest(TestCase):
    def test_adj_indices_from_faces(self):
        faces = torch.LongTensor([[2, 3, 0], [1, 0, 2]])
        edges = edges_from_faces(faces)
        # print(edges)
