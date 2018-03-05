from unittest import TestCase

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from ..modules.pooling import GraclusMaxPool


class PoolingTest(TestCase):
    def test_init(self):
        pool = GraclusMaxPool(4)
        self.assertEqual(pool.level, 4)
        self.assertEqual(pool.kernel_size, 16)
        self.assertEqual(pool.__repr__(), 'GraclusMaxPool(level=4)')

        pool = GraclusMaxPool()
        self.assertEqual(pool.level, 1)
        self.assertEqual(pool.kernel_size, 2)
        self.assertEqual(pool.__repr__(), 'GraclusMaxPool(level=1)')

    def test_graclus_max_pool(self):
        input = torch.FloatTensor([[1, 4], [3, 2], [4, 7], [6, 5]])
        input = Variable(input)
        output = GraclusMaxPool(1)(input)

        expected_output = [[3, 4], [6, 7]]
        assert_equal(output.data.numpy(), expected_output)
