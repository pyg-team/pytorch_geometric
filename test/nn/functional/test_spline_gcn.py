from unittest import TestCase

import torch
# from torch.autograd import Variable
from numpy.testing import assert_equal

from torch_geometric.nn.functional.spline_gcn import closed_spline


class SplineGcnTest(TestCase):
    def test_forward(self):
        pass
        # i = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        # v1 = torch.FloatTensor([1, 2, 3, 4, 5, 6])
        # v2 = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
        #                         [11, 12]])
        # a = torch.sparse.FloatTensor(i, v2, torch.Size([3, 3, 2]))

    def test_closed_spline(self):
        values = torch.FloatTensor([
            0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
            0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1
        ])

        B, C = closed_spline(values, partitions=4)
        B, C = B.numpy(), C.numpy()

        assert_equal(B, [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0, 1],
                         [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0, 1],
                         [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0, 1],
                         [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0, 1]])
        assert_equal(C, [[0, 3], [0, 3], [0, 3], [1, 0], [1, 0], [1, 0],
                         [1, 0], [2, 1], [2, 1], [2, 1], [2, 1], [3, 2],
                         [3, 2], [3, 2], [3, 2], [0, 3]])

    def test_open_spline(self):
        pass
