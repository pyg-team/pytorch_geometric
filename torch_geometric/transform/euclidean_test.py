from unittest import TestCase
import torch
from numpy.testing import assert_equal

from .euclidean import EuclideanAdj


class EuclideanTest(TestCase):
    def test_euclidean_adj(self):
        vertices = torch.LongTensor([[1, 0], [0, 0], [-2, 4]])
        edges = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        transform = EuclideanAdj()

        vertices, adj = transform((vertices, edges))
        adj = adj.to_dense()

        expected_vertices = [[1, 0], [0, 0], [-2, 4]]
        expected_adj_x = [
            [0., 0.375, 0],
            [0.625, 0, 0.25],
            [0, 0.75, 0],
        ]
        expected_adj_y = [
            [0., 0.5, 0],
            [0.5, 0, 1],
            [0, 0, 0],
        ]

        assert_equal(vertices.numpy(), expected_vertices)
        assert_equal(adj[:, :, 0].numpy(), expected_adj_x)
        assert_equal(adj[:, :, 1].numpy(), expected_adj_y)
