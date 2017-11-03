from unittest import TestCase
import torch
from numpy.testing import assert_equal, assert_almost_equal

from .graclus import Graclus


class GraclusTest(TestCase):
    def test_graclus_adj(self):
        input = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        position = torch.FloatTensor([[1, 1], [3, 2], [3, 0], [4, 1], [8, 3]])
        index = torch.LongTensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 3, 4],
            [1, 2, 0, 3, 0, 3, 1, 2, 4, 3],
        ])
        weight = torch.FloatTensor(10).fill_(1)
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([5, 5]))

        transform = Graclus()
        rid = torch.LongTensor([4, 3, 2, 0, 1])

        input, adj, position = transform((input, adj, position), rid)

        # Cluster = [1, 0, 1, 0, 2, 2].
        expected_input = [[3, 4], [7, 8], [1, 2], [5, 6], [9, 10], [0, 0]]
        assert_equal(input.numpy(), expected_input)

        self.assertEqual(len(position), 2)
        expected_position_0 = [[3, 2], [4, 1], [1, 1], [3, 0], [8, 3], [0, 0]]
        assert_equal(position[0].numpy(), expected_position_0)
        expected_position_1 = [[3.5, 1.5], [2, 0.5], [8, 3]]
        assert_equal(position[1].numpy(), expected_position_1)

        expected_adj_0 = [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        assert_equal(adj[0].to_dense().numpy(), expected_adj_0)
        expected_adj_1 = [
            [0, 2, 1],
            [2, 0, 0],
            [1, 0, 0],
        ]
        assert_equal(adj[1].to_dense().numpy(), expected_adj_1)
