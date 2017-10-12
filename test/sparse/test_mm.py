from unittest import TestCase

import torch
from torch.autograd import Variable
from numpy.testing import assert_equal

from torch_geometric.sparse import mm


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

    def test_autograd(self):
        i = torch.LongTensor([[0, 2], [1, 0]])
        v = torch.FloatTensor([4, 3])
        a_1 = torch.sparse.FloatTensor(i.t(), v, torch.Size([3, 3]))
        A_1 = Variable(a_1, requires_grad=True)
        a_2 = torch.FloatTensor([[0, 0, 4], [3, 0, 0], [0, 0, 0]])
        A_2 = Variable(a_2, requires_grad=True)
        f_1 = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
        F_1 = Variable(f_1, requires_grad=True)
        f_2 = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
        F_2 = Variable(f_2, requires_grad=True)

        C_1 = mm(A_1, F_1)
        C_2 = torch.mm(A_2, F_2)
        out_1 = C_1.mean()
        out_2 = C_2.mean()
        out_1.backward()
        out_2.backward()

        assert_equal(a_1.to_dense().numpy(), a_2.numpy())
        assert_equal(C_1.data.numpy(), C_2.data.numpy())
        assert_equal(A_1.grad.data.numpy(), A_2.grad.data.numpy())
        assert_equal(F_1.grad.data.numpy(), F_2.grad.data.numpy())
