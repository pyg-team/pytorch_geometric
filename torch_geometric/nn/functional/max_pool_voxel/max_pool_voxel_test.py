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
        weight = torch.ones(12).cuda()
        adj = torch.cuda.sparse.FloatTensor(index, weight, torch.Size([6, 6]))
        position = [[-2, -2], [2, 2], [-1, -1], [1, 1], [-2, 2], [2, -2]]
        position = torch.cuda.FloatTensor(position)

        max_pool = MaxPoolVoxel(adj, position, cluster_size, 4)

        output, adj, position = max_pool(input)

        expected_output = [[3, 3], [-2, -1], [-1, -2], [2, 5]]
        assert_equal(output.data.cpu().numpy(), expected_output)

        expected_adj = [[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0]]
        assert_equal(adj.data.to_dense().cpu().numpy(), expected_adj)

        expected_position = [[-1.5, -1.5], [2, -2], [-2, 2], [1.5, 1.5]]
        assert_equal(position.data.cpu().numpy(), expected_position)

        grad_output = torch.cuda.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        output.backward(grad_output)

        expected_grad = [[0, 2], [7, 0], [1, 0], [0, 8], [5, 6], [3, 4]]
        assert_equal(input.grad.data.cpu().numpy(), expected_grad)
