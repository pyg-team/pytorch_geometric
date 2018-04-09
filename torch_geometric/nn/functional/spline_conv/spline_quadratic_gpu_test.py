# import unittest

# import torch
# from numpy.testing import assert_equal, assert_almost_equal

# if torch.cuda.is_available():
#     from .compute_spline_basis import compute_spline_basis
#     from .compute_spline_basis import get_basis_kernel

# class SplineQuadraticGPUTest(unittest.TestCase):
#     @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
#     def test_open_spline(self):
#         input = torch.cuda.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
#         kernel_size = torch.cuda.LongTensor([6])
#         is_open_spline = torch.cuda.LongTensor([1])
#         k_max = 3
#         K = 6
#         dim = 1
#         basis_kernel = get_basis_kernel(k_max, K, dim, 2)

#         a1, i1 = compute_spline_basis(input, kernel_size, is_open_spline, 6,
#                                       basis_kernel)

#         a2 = [[0.5, 0.5, 0], [0.32, 0.66, 0.02], [0.5, 0.5, 0],
#              [0.5, 0.5, 0], [0.5, 0.5, 0], [0.02, 0.66, 0.32], [0.5, 0.5, 0]]
#         i2 = [[0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
#               [3, 4, 5], [4, 5, 0]]

#         assert_almost_equal(a1.cpu().numpy(), a2, 2)
#         assert_equal(i1.cpu().numpy(), i2)

#     @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
#     def test_closed_spline(self):
#         input = torch.cuda.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
#         kernel_size = torch.cuda.LongTensor([4])
#         is_open_spline = torch.cuda.LongTensor([0])
#         k_max = 3
#         K = 4
#         dim = 1
#         basis_kernel = get_basis_kernel(k_max, K, dim, 2)
#         a1, i1 = compute_spline_basis(input, kernel_size, is_open_spline, 4,
#                                       basis_kernel)

#         a2 = [[0.5, 0.5, 0], [0.32, 0.66, 0.02], [0.5, 0.5, 0],
#              [0.5, 0.5, 0], [0.5, 0.5, 0], [0.02, 0.66, 0.32], [0.5, 0.5, 0]]
#         i2 = [[0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1],
#               [3, 0, 1], [0, 1, 2]]

#         assert_almost_equal(a1.cpu().numpy(), a2, 2)
#         assert_equal(i1.cpu().numpy(), i2)
