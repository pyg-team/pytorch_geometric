from unittest import TestCase
import torch
from torch.autograd import Variable

from numpy.testing import assert_almost_equal

from .lin import lin


class LinTest(TestCase):
    def test_forward(self):
        features = Variable(torch.FloatTensor([
            [1, 2],
            [3, 4],
            [5, 6],
        ]))
        weight = Variable(
            torch.FloatTensor([
                [[0.1], [0.5]],
                [[0.2], [0.6]],
                [[0.3], [0.7]],
                [[0.4], [0.8]],
            ]))
        bias = Variable(torch.FloatTensor([1, 2, 3, 4]))

        out = lin(features, weight)
        expected = [
            [1.1, 1.4, 1.7, 2.0],
            [2.3, 3.0, 3.7, 4.4],
            [3.5, 4.6, 5.7, 6.8],
        ]
        assert_almost_equal(out.data.numpy(), expected, 1)

        out = lin(features, weight, bias)
        expected = [
            [2.1, 3.4, 4.7, 6.0],
            [3.3, 5.0, 6.7, 8.4],
            [4.5, 6.6, 8.7, 10.8],
        ]
        assert_almost_equal(out.data.numpy(), expected, 1)

    def test_backward(self):
        pass
