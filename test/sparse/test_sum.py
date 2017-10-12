from unittest import TestCase

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from torch_geometric.sparse import sum


class SumTest(TestCase):
    def test_tensor(self):
        i = torch.LongTensor([[0, 2], [1, 0]])
        v = torch.FloatTensor([4, 3])
        a_1 = torch.sparse.FloatTensor(i.t(), v, torch.Size([3, 3]))
        a_2 = torch.FloatTensor([[0, 0, 4], [3, 0, 0], [0, 0, 0]])

        assert_equal(a_1.to_dense().numpy(), a_2.numpy())
        assert_equal(sum(a_1), torch.sum(a_2))
        assert_equal(sum(a_1, dim=0).numpy(), torch.sum(a_2, dim=0).numpy())
        assert_equal(sum(a_1, dim=1).numpy(), torch.sum(a_2, dim=1).numpy())
        assert_equal(sum(a_1, dim=1).numpy(), torch.sum(a_2, dim=1).numpy())
        assert_equal(
            sum(a_1, dim=0, keepdim=True).numpy(),
            torch.sum(a_2, dim=0, keepdim=True).numpy())
        assert_equal(
            sum(a_1, dim=1, keepdim=True).numpy(),
            torch.sum(a_2, dim=1, keepdim=True).numpy())
        assert_equal(
            sum(a_1, dim=0, keepdim=False).numpy(),
            torch.sum(a_2, dim=0, keepdim=False).numpy())
        assert_equal(
            sum(a_1, dim=1, keepdim=False).numpy(),
            torch.sum(a_2, dim=1, keepdim=False).numpy())

    def test_autograd(self):
        pass
