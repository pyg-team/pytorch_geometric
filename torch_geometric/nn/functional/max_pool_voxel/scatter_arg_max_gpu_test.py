import unittest
import torch
from numpy.testing import assert_equal

if torch.cuda.is_available():
    from .scatter_arg_max_gpu import scatter_arg_max_gpu


class ScatterArgMaxGPUTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_scatter_arg_max(self):
        input = [[-1, 3], [-3, 5], [4, -5], [2, 1], [-1, -2]]
        input = torch.cuda.FloatTensor(input)
        cluster = torch.cuda.IntTensor([0, 1, 1, 0, 2])
        max, argmax = scatter_arg_max_gpu(input, cluster, cluster.max() + 1)

        expected_max = [[2, 3], [4, 5], [-1, -2]]
        assert_equal(max.cpu().numpy(), expected_max)
