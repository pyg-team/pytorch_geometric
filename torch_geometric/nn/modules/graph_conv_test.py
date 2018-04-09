from unittest import TestCase

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from .graph_conv import GraphConv
from ..functional.graph_conv import graph_conv


class GraphConvTest(TestCase):
    def test_init(self):
        conv = GraphConv(1, 10)
        self.assertEqual(conv.in_features, 1)
        self.assertEqual(conv.out_features, 10)
        self.assertEqual(conv.weight.data.size(), (1, 10))
        self.assertEqual(conv.bias.data.size(), (10, ))
        self.assertEqual(conv.__repr__(), 'GraphConv(1, 10)')

        conv = GraphConv(1, 10, bias=False)
        self.assertIsNone(conv.bias)
        self.assertEqual(conv.__repr__(), 'GraphConv(1, 10, bias=False)')

    def test_forward(self):
        conv = GraphConv(1, 10)
        x = Variable(torch.Tensor([[1], [2], [3]]))
        i = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        w = torch.Tensor([1, 1, 1, 1, 1, 1])

        out = conv(x, i, w)
        expected_out = graph_conv(x, i, w, conv.weight, conv.bias)

        assert_equal(out.data.numpy(), expected_out.data.numpy())
