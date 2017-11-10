from unittest import TestCase

import torch
from numpy.testing import assert_equal

from .point_cloud import nn_graph


class PointCloudTest(TestCase):
    def test_nn_graph(self):
        point = torch.FloatTensor([[0, 0], [1, 1], [2, 0], [1, -2], [-1, 0]])
        index = nn_graph(point, k=2)

        expected_index = [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                          [4, 1, 0, 2, 1, 0, 0, 2, 0, 1]]

        assert_equal(index.numpy(), expected_index)
