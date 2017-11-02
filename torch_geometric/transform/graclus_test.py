from unittest import TestCase
import torch
# from numpy.testing import assert_equal, assert_almost_equal

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
        rid = torch.LongTensor([0, 1, 2, 3, 4])
        rid = torch.LongTensor([4, 3, 2, 1, 0])

        input, adj, position = transform((input, adj, position), rid)

        # print(input)
        # print(position)
        # print(adj[0].to_dense())
        # print(adj[1].to_dense())
