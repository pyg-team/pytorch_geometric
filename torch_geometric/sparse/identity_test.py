from unittest import TestCase

import torch
from numpy.testing import assert_equal

from .identity import identity


class IdentityTest(TestCase):
    def test_identity_n_n(self):
        out = identity(torch.Size([3, 3]))
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

        assert_equal(out.to_dense().numpy(), expected)

    def test_identity_n_m(self):
        out = identity(torch.Size([3, 4]))
        expected = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]

        assert_equal(out.to_dense().numpy(), expected)

    def test_identity_m_n(self):
        out = identity(torch.Size([4, 3]))
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]

        assert_equal(out.to_dense().numpy(), expected)
