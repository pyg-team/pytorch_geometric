from unittest import TestCase

import torch
from numpy.testing import assert_equal

from ..modules.pooling import GraclusMaxPool


class PoolingTest(TestCase):
    def test_graclus_max_pool(self):
        input = torch.FloatTensor([[1, 4], [3, 2], [4, 7], [6, 5]])
        output = GraclusMaxPool(1)(input)

        expected_output = [[3, 4], [6, 7]]
        assert_equal(output.data.numpy(), expected_output)
