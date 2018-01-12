from unittest import TestCase

import torch
from torch.autograd import Variable, gradcheck

from .pool import max_pool, avg_pool, _max_pool, _avg_pool


def max_pool_backward(input):
    cluster = torch.LongTensor([0, 1, 1, 0])
    return _max_pool(input, cluster)


def avg_pool_backward(input):
    cluster = torch.LongTensor([0, 1, 1, 0])
    return _avg_pool(input, cluster)


class PoolTest(TestCase):
    def test_forward(self):
        input = torch.FloatTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        index = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        position = torch.FloatTensor([[-2, 2], [2, 2], [2, -2], [-2, -2]])
        cluster = torch.LongTensor([0, 1, 1, 0])

        expected_max_input = [[5, 3], [8, 7]]
        expected_average_input = [[3, 2.5], [6, 5]]
        expected_index = [[0, 1], [1, 0]]
        expected_position = [[-2, 0], [2, 0]]

        input, index, position = max_pool(input, index, position, cluster)

        input, index, position = avg_pool(input, index, position, cluster)

    def test_backward(self):
        input = torch.DoubleTensor([[5, 2], [4, 7], [8, 3], [1, 3]])
        input = Variable(input, requires_grad=True)

        op = max_pool_backward
        test = gradcheck(op, (input,), eps=1e-6, atol=1e-4)
        self.assertTrue(test)

        op = avg_pool_backward
        test = gradcheck(op, (input,), eps=1e-6, atol=1e-4)
        self.assertTrue(test)
