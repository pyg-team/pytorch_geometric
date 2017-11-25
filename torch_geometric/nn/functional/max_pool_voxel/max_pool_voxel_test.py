import unittest
import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

if torch.cuda.is_available():
    from .max_pool_voxel import MaxPoolVoxel


class MaxPoolVoxelGPUTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_max_pool_voxel(self):
        cluster_size = torch.cuda.LongTensor([2, 2])
        input = [[1, 3], [2, 4], [3, 2], [1, 5], [-1, -2], [-2, -1]]
        input = Variable(torch.cuda.FloatTensor(input), requires_grad=True)
        row = [0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 5]
        col = [2, 3, 0, 3, 4, 5, 1, 2, 5, 2, 2, 3]
        index = torch.cuda.LongTensor([row, col])
        weight = torch.cuda.ones(12)
        adj = torch.cuda.sparse.FloatTensor(index, weight, torch.Size([6, 6]))
        position = [[-2, -2], [2, 2], [-1, -1], [1, 1], [-2, 2], [2, -2]]
        position = torch.cuda.FloatTensor(position)

        max_pool = MaxPoolVoxel(cluster_size, 4)

        output = max_pool(input, adj, position)
        print(output)

        expected_output = [[3, 3], [2, 5], [-1, -2], [-2, -1]]
        assert_equal(output.data.cpu().numpy(), expected_output)
