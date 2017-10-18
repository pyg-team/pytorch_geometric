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
        out, slices = stack([mat1, mat2], horizontal=True, vertical=False)

        expected_out = [
            [0, 1, 0, 0, 0, 3, 0],
            [2, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        assert_equal(out.to_dense().numpy(), expected_out)

        expected_slices = [4, 7]
        assert_equal(slices.numpy(), expected_slices)

    def test_stack_vertical(self):
        out, slices = stack([mat1, mat2], horizontal=False, vertical=True)
        expected = [
            [0, 1, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 3, 0, 0],
            [4, 0, 0, 0],
        ]
        assert_equal(out.to_dense().numpy(), expected)

        expected_slices = [3, 5]
        assert_equal(slices.numpy(), expected_slices)

    def test_stack_diagonal(self):
        out, slices = stack([mat1, mat2], horizontal=True, vertical=True)
        expected = [
            [0, 1, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 4, 0, 0],
        ]
        assert_equal(out.to_dense().numpy(), expected)

        expected_slices = [[3, 4], [5, 7]]
        assert_equal(slices.numpy(), expected_slices)

    def test_dynamic_type(self):
        print('TODO: test dynamic type')

    def test_stack_dense_sparse_matrices(self):
        print('TODO: test dense sparse type')
