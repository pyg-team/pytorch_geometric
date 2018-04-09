from __future__ import division

import unittest
import torch
from torch.autograd import Variable, gradcheck

if torch.cuda.is_available():
    from .spline_conv_gpu import (get_basis_kernel, get_basis_backward_kernel,
                                  get_weighting_forward_kernel,
                                  get_weighting_backward_kernel, SplineConvGPU)


class SplineConvTest(unittest.TestCase):
    '''
    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_forward_gpu(self):
        edges = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        values = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
        values = torch.FloatTensor(values).double()
        adj = {'indices': edges.cuda(), 'values': Variable(values.cuda()),
               'size': torch.Size([5, 5, 2])}


        kernel_size = torch.cuda.LongTensor([3, 4])
        is_open_spline = torch.cuda.LongTensor([1, 0])

        input = torch.FloatTensor([[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
        input = input.double()
        weight = torch.arange(0.5, 0.5 * 27, step=0.5).view(13, 2, 1).double()
        input, weight = input.cuda(), weight.cuda()
        input, weight = Variable(input), Variable(weight)
        row, col = adj['indices']
        output = input[col]
        K = 12
        in_features = 2
        out_features = 1
        degree = 1
        dim = 2
        k_max = (degree+1)**dim
        fw_k = get_weighting_forward_kernel(in_features, out_features, k_max)
        bw_k = get_weighting_backward_kernel(in_features, out_features, k_max,
                                             K, True)

        basis_fw_k = get_basis_kernel(k_max, K, dim, degree)

        basis_bw_k = get_basis_backward_kernel(k_max, K, dim, degree)


        #output = spline_conv(
        #    adj, input, weight, kernel_size, is_open_spline, K, fw_k, bw_k,
        #    basis_fw_k, basis_bw_k,bp_to_adj=True)
        values = adj['values']
        output = SplineConvGPU(kernel_size, is_open_spline, K, degree,
                           basis_fw_k, basis_bw_k, fw_k, bw_k, bp_to_adj=True)\
            (output, weight, values)

        zero = output.data.new(adj['size'][1], output.size(1)).fill_(0.0)
        zero = Variable(zero) if not torch.is_tensor(output) else zero
        r = row.view(-1, 1).expand(row.size(0), output.size(1))
        output = zero.scatter_add_(0, Variable(r), output)

        # Weighten root node features by multiplying with root weight.
        output += torch.mm(input, weight[-1])

        # Normalize output by degree.
        ones = values.data.new(values.size(0)).fill_(1)
        zero = values.data.new(output.size(0)).fill_(0)
        degree = zero.scatter_add_(0, row, ones)
        degree = torch.clamp(degree, min=1)
        output = output / Variable(degree.view(-1, 1))


        expected_output = [
            [(12.5 * 9 + 13 * 10 + 266) / 4],
            [12.5 * 1 + 13 * 2],
            [12.5 * 3 + 13 * 4],
            [12.5 * 5 + 13 * 6],
            [12.5 * 7 + 13 * 8],
        ]
        assert_almost_equal(output.cpu().data.numpy(), expected_output, 1)


    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_backward(self):
        kernel_size = torch.cuda.LongTensor([3, 4])
        is_open_spline = torch.cuda.LongTensor([1, 1])

        input = torch.randn(4, 2).double().cuda()
        weight = torch.randn(12, 2, 1).double().cuda()
        values = torch.FloatTensor(4, 2).uniform_(0, 1).double().cuda()
        print(values)
        input = Variable(input, requires_grad=True)
        weight = Variable(weight, requires_grad=True)
        values = Variable(values, requires_grad=True)

        K = 12
        in_features = 2
        out_features = 1
        degree = 1
        dim = 2
        k_max = (degree + 1) ** dim
        fw_k = get_weighting_forward_kernel(in_features, out_features, k_max)
        bw_k = get_weighting_backward_kernel(in_features, out_features, k_max,
                                             K, bp_to_adj=True)

        basis_fw_k = get_basis_kernel(k_max, K, dim, degree)

        basis_bw_k = get_basis_backward_kernel(k_max, K, dim, degree)

        op = SplineConvGPU(kernel_size, is_open_spline, K, degree,
                           basis_fw_k, basis_bw_k, fw_k, bw_k, bp_to_adj=True)
        print(op(input, weight, values))
        #test = gradcheck(op, (input, weight, values), eps=1e-6, atol=1e-4)

        #self.assertTrue(test)
    '''

    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_backward(self):

        input = torch.randn(4, 2).double().cuda()
        weight = torch.randn(9, 2, 1).double().cuda()
        values = torch.FloatTensor(4, 2).uniform_(0, 1).double().cuda()
        input = Variable(input, requires_grad=True)
        weight = Variable(weight, requires_grad=True)
        values = Variable(values, requires_grad=True)

        K = 9
        in_features = 2
        out_features = 1
        degree = 1
        dim = 2
        k_max = (degree + 1)**dim
        kernel_size = torch.cuda.LongTensor([3, 3])
        is_open_spline = torch.cuda.LongTensor([1, 0])
        fw_k = get_weighting_forward_kernel(
            in_features, out_features, k_max, dtype='double')
        bw_k = get_weighting_backward_kernel(
            in_features, out_features, k_max, K, True, dtype='double')

        basis_fw_k = get_basis_kernel(k_max, K, dim, degree, dtype='double')

        basis_bw_k = get_basis_backward_kernel(
            k_max, K, dim, degree, dtype='double')

        op = SplineConvGPU(
            kernel_size,
            is_open_spline,
            K,
            degree,
            basis_fw_k,
            basis_bw_k,
            fw_k,
            bw_k,
            bp_to_adj=True)
        test = gradcheck(op, (input, weight, values), eps=1e-6, atol=1e-4)

        self.assertTrue(test)
