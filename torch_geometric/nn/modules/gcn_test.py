from unittest import TestCase

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from .gcn import GCN
from ..functional.gcn import gcn


class GCNTest(TestCase):
    def test_init(self):
        conv = GCN(1, 10)
        self.assertEqual(conv.in_features, 1)
        self.assertEqual(conv.out_features, 10)
        self.assertEqual(conv.weight.data.size(), (1, 10))
        self.assertEqual(conv.bias.data.size(), (10, ))
        self.assertEqual(conv.__repr__(), 'GCN(1, 10)')

        conv = GCN(1, 10, bias=False)
        self.assertIsNone(conv.bias)
        self.assertEqual(conv.__repr__(), 'GCN(1, 10, bias=False)')

    def test_forward(self):
        conv = GCN(1, 10)
        i = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        w = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        adj = torch.sparse.FloatTensor(i, w, torch.Size([3, 3]))
        features = Variable(torch.FloatTensor([[1], [2], [3]]))

        out = conv(adj, features)
        expected_out = gcn(adj, features, conv.weight, conv.bias)

        assert_equal(out.data.numpy(), expected_out.data.numpy())
        assert_equal(out.data.numpy(), expected_out.data.numpy())
