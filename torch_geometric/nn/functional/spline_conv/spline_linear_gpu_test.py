import unittest

import torch
from numpy.testing import assert_equal, assert_almost_equal

if torch.cuda.is_available():
    from .compute_spline_basis import compute_spline_basis
    from .compute_spline_basis import get_basis_kernel


class SplineLinearGPUTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_open_spline(self):
        input = torch.cuda.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.cuda.LongTensor([5])
        is_open_spline = torch.cuda.LongTensor([1])
        k_max = 2
        K = 5
        dim = 1
        basis_kernel = get_basis_kernel(k_max, K, dim, 1)
        a1, i1 = compute_spline_basis(input, kernel_size, is_open_spline, 5,
                                      basis_kernel)

        a2 = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i2 = [[1, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 3], [0, 4]]

        assert_almost_equal(a1.cpu().numpy(), a2, 2)
        assert_equal(i1.cpu().numpy(), i2)

    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_closed_spline(self):
        input = torch.cuda.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.cuda.LongTensor([4])
        is_open_spline = torch.cuda.LongTensor([0])
        k_max = 2
        K = 4
        dim = 1
        basis_kernel = get_basis_kernel(k_max, K, dim, 1)
        a1, i1 = compute_spline_basis(input, kernel_size, is_open_spline, 4,
                                      basis_kernel)

        a2 = [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [0, 1], [0.8, 0.2], [0, 1]]
        i2 = [[1, 0], [1, 0], [2, 1], [3, 2], [0, 3], [0, 3], [1, 0]]

        assert_almost_equal(a1.cpu().numpy(), a2, 2)
        assert_equal(i1.cpu().numpy(), i2)
