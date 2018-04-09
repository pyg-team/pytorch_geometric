import unittest

import torch
from torch.autograd import Variable, gradcheck
from numpy.testing import assert_equal

from .pool import max_pool, avg_pool, _max_pool, _avg_pool


def max_pool_backward(input):
    cluster = torch.LongTensor([0, 1, 1, 0])
    cluster = cluster.cuda() if input.is_cuda else cluster
    return _max_pool(input, cluster, None)


def avg_pool_backward(input):
    cluster = torch.LongTensor([0, 1, 1, 0])
    cluster = cluster.cuda() if input.is_cuda else cluster
    return _avg_pool(input, cluster, None, None)


class PoolTest(unittest.TestCase):
    def test_max_forward_cpu(self):
        input = torch.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        index = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.LongTensor([0, 1, 1, 0])

        output, index, position = max_pool(input, index, position, cluster)

        expected_output = [[5, 3], [8, 7]]
        expected_index = [[0, 1], [1, 0]]
        expected_position = [[-2, 0], [2, 0]]

        assert_equal(output.tolist(), expected_output)
        assert_equal(index.tolist(), expected_index)
        assert_equal(position.tolist(), expected_position)

    def test_avg_forward_cpu(self):
        input = torch.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        index = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.LongTensor([0, 1, 1, 0])

        output, index, position = avg_pool(input, index, position, cluster)

        expected_output = [[3, 2.5], [6, 5]]
        expected_index = [[0, 1], [1, 0]]
        expected_position = [[-2, 0], [2, 0]]

        assert_equal(output.tolist(), expected_output)
        assert_equal(index.tolist(), expected_index)
        assert_equal(position.tolist(), expected_position)

    def test_max_backward_cpu(self):
        input = torch.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        input = Variable(input, requires_grad=True)
        index = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.LongTensor([0, 1, 1, 0])

        output, index, position = max_pool(input, index, position, cluster)
        output.backward(torch.FloatTensor([[10, 20], [5, 15]]))

        expected_grad = [[10, 0], [0, 15], [5, 0], [0, 20]]
        assert_equal(input.grad.data.tolist(), expected_grad)

    def test_avg_backward_cpu(self):
        input = torch.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        input = Variable(input, requires_grad=True)
        index = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.LongTensor([0, 1, 1, 0])

        output, index, position = avg_pool(input, index, position, cluster)
        output.backward(torch.FloatTensor([[10, 20], [5, 15]]))

        expected_grad = [[5, 10], [2.5, 7.5], [2.5, 7.5], [5, 10]]
        assert_equal(input.grad.data.tolist(), expected_grad)

    def test_auto_backward(self):
        input = torch.DoubleTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        input = Variable(input, requires_grad=True)

        op = max_pool_backward
        test = gradcheck(op, (input, ), eps=1e-6, atol=1e-4)
        self.assertTrue(test)

        op = avg_pool_backward
        test = gradcheck(op, (input, ), eps=1e-6, atol=1e-4)
        self.assertTrue(test)

    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_max_forward_gpu(self):
        input = torch.cuda.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        index = torch.cuda.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.cuda.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.cuda.LongTensor([0, 1, 1, 0])

        output, index, position = max_pool(input, index, position, cluster)

        expected_output = [[5, 3], [8, 7]]
        expected_index = [[0, 1], [1, 0]]
        expected_position = [[-2, 0], [2, 0]]

        assert_equal(output.cpu().tolist(), expected_output)
        assert_equal(index.cpu().tolist(), expected_index)
        assert_equal(position.cpu().tolist(), expected_position)

    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_avg_forward_gpu(self):
        input = torch.cuda.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        index = torch.cuda.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.cuda.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.cuda.LongTensor([0, 1, 1, 0])

        output, index, position = avg_pool(input, index, position, cluster)

        expected_output = [[3, 2.5], [6, 5]]
        expected_index = [[0, 1], [1, 0]]
        expected_position = [[-2, 0], [2, 0]]

        assert_equal(output.cpu().tolist(), expected_output)
        assert_equal(index.cpu().tolist(), expected_index)
        assert_equal(position.cpu().tolist(), expected_position)
