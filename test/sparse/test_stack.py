from unittest import TestCase
import torch
from numpy.testing import assert_equal

from torch_geometric.sparse.stack import stack

mat1 = torch.sparse.FloatTensor(
    torch.LongTensor([[0, 1], [1, 0]]),
    torch.FloatTensor([1, 2]), torch.Size([3, 4]))

mat2 = torch.sparse.FloatTensor(
    torch.LongTensor([[0, 1], [1, 0]]),
    torch.FloatTensor([3, 4]), torch.Size([2, 3]))


class StackTest(TestCase):
    def test_stack_horizontal(self):
        out = stack([mat1, mat2], horizontal=True, vertical=False).to_dense()
        expected = [
            [0, 1, 0, 0, 0, 3, 0],
            [2, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        assert_equal(out.numpy(), expected)

    def test_stack_vertical(self):
        out = stack([mat1, mat2], horizontal=False, vertical=True).to_dense()
        print(out)
        expected = [
            [0, 1, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 3, 0, 0],
            [4, 0, 0, 0],
        ]
        assert_equal(out.numpy(), expected)

    def test_stack_diagonal(self):
        out = stack([mat1, mat2], horizontal=True, vertical=True).to_dense()
        expected = [
            [0, 1, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 4, 0, 0],
        ]
        assert_equal(out.numpy(), expected)

    def test_dynamic_type(self):
        print('TODO: test dynamic type')
