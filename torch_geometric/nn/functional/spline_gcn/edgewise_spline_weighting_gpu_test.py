from unittest import TestCase

import torch
from torch.autograd import Variable, gradcheck
from numpy.testing import assert_equal

from .spline import spline

if torch.cuda.is_available():
    from .edgewise_spline_weighting_gpu import EdgewiseSplineWeightingGPU


class EdgewiseSplineWeightingGPUTest(TestCase):
    def test_forward(self):
        if not torch.cuda.is_available():
            return

        input = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
        input = torch.FloatTensor(input)
        kernel_size = torch.LongTensor([3, 4])
        is_open_spline = torch.LongTensor([1, 0])

        amount, index = spline(input, kernel_size, is_open_spline, 1)
        amount, index = amount.cuda(), index.cuda()

        input = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)
        input, weight = input.cuda(), weight.cuda()
        input, weight = Variable(input), Variable(weight)

        op = EdgewiseSplineWeightingGPU(amount, index)
        out = op(input, weight)

        expected_out = [
            [0.25 * (1 * (0.5 + 1.5 + 4.5 + 5.5) + 2 * (1 + 2 + 5 + 6))],
            [0.25 * (3 * (1.5 + 2.5 + 5.5 + 6.5) + 4 * (2 + 3 + 6 + 7))],
            [0.25 * (5 * (6.5 + 7.5 + 10.5 + 11.5) + 6 * (7 + 8 + 11 + 12))],
            [0.25 * (7 * (4.5 + 7.5 + 8.5 + 11.5) + 8 * (5 + 8 + 9 + 12))],
        ]

        assert_equal(out.cpu().data.numpy(), expected_out)

    def test_backward(self):
        if not torch.cuda.is_available():
            return

        input = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
        input = torch.DoubleTensor(input)
        kernel_size = torch.LongTensor([3, 4])
        is_open_spline = torch.LongTensor([1, 0])

        amount, index = spline(input, kernel_size, is_open_spline, 1)
        amount, index = amount.cuda(), index.cuda()

        input = torch.randn(4, 2).double()
        weight = torch.randn(12, 2, 1).double()
        input, weight = input.cuda(), weight.cuda()
        input = Variable(input, requires_grad=True)
        weight = Variable(weight, requires_grad=True)

        op = EdgewiseSplineWeightingGPU(amount, index)
        test = gradcheck(op, (input, weight), eps=1e-6, atol=1e-4)
        self.assertTrue(test)
