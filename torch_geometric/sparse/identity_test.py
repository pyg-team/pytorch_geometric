from unittest import TestCase
import torch
from numpy.testing import assert_equal

from torch_geometric.sparse.identity import identity


class IdentityTest(TestCase):
    def test_identity(self):
        out = identity(torch.Size([3, 4]))
        expected = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]

        assert_equal(out.to_dense().numpy(), expected)

    def test_dynamic_type(self):
        print('TODO: test dynamic type')

    def test_stack_dense_sparse_matrices(self):
        print('TODO: test dense sparse type')
