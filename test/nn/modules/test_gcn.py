from unittest import TestCase

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from torch_geometric.nn.modules import GCN
from torch_geometric.nn._functions import gcn as _gcn


class GCNTest(TestCase):
    def test_init(self):
        gcn = GCN(1, 10)
        self.assertEqual(gcn.in_features, 1)
        self.assertEqual(gcn.out_features, 10)
        self.assertEqual(gcn.weight.data.size(), (1, 10))
        self.assertEqual(gcn.bias.data.size(), (10, ))
        self.assertEqual(gcn.__repr__(), 'GCN(1, 10)')

        gcn = GCN(1, 10, bias=False)
        self.assertIsNone(gcn.bias)
        self.assertEqual(gcn.__repr__(), 'GCN(1, 10, bias=False)')

    def test_forward(self):
        gcn = GCN(1, 10)
        i = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        w = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        adj = Variable(torch.sparse.FloatTensor(i, w, torch.Size([3, 3])))
        features = Variable(torch.FloatTensor([[1], [2], [3]]))

        out = gcn(adj, features)
        expected_out = _gcn(adj, features, gcn.weight, gcn.bias)

        assert_equal(out.data.numpy(), expected_out.data.numpy())
        assert_equal(out.data.numpy(), expected_out.data.numpy())
