from unittest import TestCase

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from .sum import sum


class SumTest(TestCase):
    def test_tensor_2d(self):
        i = torch.LongTensor([[0, 0, 1], [1, 2, 0]])
        v = torch.FloatTensor([4, 3, 2])
        a_1 = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
        a_2 = torch.FloatTensor([[0, 4, 3], [2, 0, 0], [0, 0, 0]])

        assert_equal(a_1.to_dense().numpy(), a_2.numpy())
        assert_equal(sum(a_1), torch.sum(a_2))
        assert_equal(sum(a_1, dim=0).numpy(), torch.sum(a_2, dim=0).numpy())
        assert_equal(sum(a_1, dim=1).numpy(), torch.sum(a_2, dim=1).numpy())

    def test_tensor_3d(self):
        i = torch.LongTensor([[0, 0, 1], [1, 2, 0]])
        v = torch.FloatTensor([[4, 5], [3, 7], [2, 1]])
        a_1 = torch.sparse.FloatTensor(i, v, torch.Size([3, 3, 2]))
        a_2 = torch.FloatTensor([
            [[0, 0], [4, 5], [3, 7]],
            [[2, 1], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ])

        assert_equal(a_1.to_dense().numpy(), a_2.numpy())
        assert_equal(sum(a_1), torch.sum(a_2))
        assert_equal(sum(a_1, dim=0).numpy(), torch.sum(a_2, dim=0).numpy())
        assert_equal(sum(a_1, dim=1).numpy(), torch.sum(a_2, dim=1).numpy())

    def test_tensor_4d(self):
        print("TODO")

    def test_autograd(self):
        print("TODO")
        return
        i = torch.LongTensor([[0, 0, 1], [1, 2, 0]])
        v = torch.FloatTensor([4, 3, 2])
        a_1 = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
        a_1 = Variable(a_1, requires_grad=True)
        a_2 = torch.FloatTensor([[0, 4, 3], [2, 0, 0], [0, 0, 0]])
        a_2 = Variable(a_2, requires_grad=True)

        c_1 = sum(a_1, dim=1)
        c_2 = torch.sum(a_2, dim=1)

        out_1 = c_1.mean()
        out_2 = c_2.mean()
        out_1.backward()
        out_2.backward()

        assert_equal(a_1.to_dense().numpy(), a_2.numpy())
        assert_equal(c_1.data.numpy(), c_2.data.numpy())
        assert_equal(a_1.grad.data.numpy(), a_2.grad.data.numpy())
