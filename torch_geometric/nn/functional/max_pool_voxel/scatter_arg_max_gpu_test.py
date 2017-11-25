import unittest
import torch
from numpy.testing import assert_equal

if torch.cuda.is_available():
    from .scatter_arg_max_gpu import scatter_arg_max_gpu


class ScatterArgMaxGPUTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_scatter_max(self):
        input = [[-1, 3], [-3, 5], [4, -5], [2, 1], [-1, -2]]
        input = torch.cuda.FloatTensor(input)
        cluster = torch.cuda.LongTensor([0, 1, 1, 0, 2])
        max = torch.cuda.FloatTensor([[2, 3], [4, 5], [-1, -2]])

        arg_max = scatter_arg_max_gpu(input, cluster, max)
        expected_arg_max = [[3, 0], [2, 1], [4, 4]]

        assert_equal(arg_max.cpu().numpy(), expected_arg_max)
