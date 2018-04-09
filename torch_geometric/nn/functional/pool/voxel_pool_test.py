import unittest

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from .voxel_pool import sparse_voxel_max_pool, dense_voxel_max_pool
from ....datasets.dataset import Data


class VoxelPoolTest(unittest.TestCase):
    def test_sparse_voxel_pool(self):
        pos = torch.FloatTensor([[0, 1], [0.5, 1.5], [0, 0], [1, 0], [0, 1],
                                 [0.5, 1.5], [0, 0], [1, 0]])
        index = torch.LongTensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                                  [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
        input = torch.FloatTensor([[1, 4], [2, 3], [3, 2], [4, 1], [4, 1],
                                   [3, 2], [2, 3], [1, 4]])
        batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
        pos, index = pos.cuda(), index.cuda()
        input, batch = input.cuda(), batch.cuda()
        input = Variable(input, requires_grad=True)

        data = Data(input, pos, index, None, None, batch)
        data, cluster = sparse_voxel_max_pool(data, 1)

        expected_cluster = [1, 1, 0, 2, 4, 4, 3, 5]
        expected_pos = [[0, 0], [0.25, 1.25], [1, 0], [0, 0], [0.25, 1.25],
                        [1, 0]]
        expected_index = [[0, 0, 1, 2, 3, 3, 4, 5], [1, 2, 0, 0, 4, 5, 3, 3]]
        expected_input = [[3, 2], [2, 4], [4, 1], [2, 3], [4, 2], [1, 4]]
        expected_batch = [0, 0, 0, 1, 1, 1]

        assert_equal(cluster.tolist(), expected_cluster)
        assert_equal(data.pos.tolist(), expected_pos)
        assert_equal(data.index.tolist(), expected_index)
        assert_equal(data.input.tolist(), expected_input)
        assert_equal(data.batch.tolist(), expected_batch)

        grad = torch.FloatTensor([[1, 6], [2, 5], [3, 4], [1, 6], [2, 5],
                                  [3, 4]]).cuda()
        data.input.backward(grad)

        expected_grad = [[0, 5], [2, 0], [1, 6], [3, 4], [2, 0], [0, 5],
                         [1, 6], [3, 4]]
        assert_equal(input.grad.tolist(), expected_grad)

    def test_dense_voxel_pool(self):
        input = torch.FloatTensor([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
        pos = torch.FloatTensor([[3, 1], [4, 1], [12, 1], [6, 1], [14, 1]])
        index = torch.LongTensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        batch = torch.LongTensor([0, 0, 0, 1, 1])
        data = Data(input, pos, index, None, None, batch)
        data, cluster = dense_voxel_max_pool(data, 5, start=0)
