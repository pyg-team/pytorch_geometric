from unittest import TestCase
import torch
from numpy.testing import assert_equal

from .mm import mm


class MmTest(TestCase):
    def test_tensor(self):
        i = torch.LongTensor([[0, 2], [1, 0]])
        v = torch.FloatTensor([4, 3])
        a_1 = torch.sparse.FloatTensor(i.t(), v, torch.Size([3, 3]))
        a_2 = torch.FloatTensor([[0, 0, 4], [3, 0, 0], [0, 0, 0]])
        f = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])

        c_1 = mm(a_1, f)
        c_2 = torch.mm(a_2, f)

        assert_equal(a_1.to_dense().numpy(), a_2.numpy())
        assert_equal(c_1.numpy(), c_2.numpy())
