from unittest import TestCase

import torch
from numpy.testing import assert_equal

from .spline_cpu import spline_cpu

if torch.cuda.is_available():
    from .spline_gpu import spline_gpu


class SplineGPUTest(TestCase):
    def test_open_spline(self):
        if not torch.cuda.is_available():
            return

        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.LongTensor([5])
        is_open_spline = torch.LongTensor([1])

        a1, i1 = spline_cpu(input, kernel_size, is_open_spline, 1)

        a2, i2 = spline_gpu(input.cuda(),
                            kernel_size.cuda(), is_open_spline.cuda(), 1)

        assert_equal(a1.numpy(), a2.cpu().numpy())
        assert_equal(i1.numpy(), i2.cpu().numpy())

    def test_closed_spline(self):
        if not torch.cuda.is_available():
            return

        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        kernel_size = torch.LongTensor([4])
        is_open_spline = torch.LongTensor([0])

        a1, i1 = spline_cpu(input, kernel_size, is_open_spline, 1)

        a2, i2 = spline_gpu(input.cuda(),
                            kernel_size.cuda(), is_open_spline.cuda(), 1)

        assert_equal(a1.numpy(), a2.cpu().numpy())
        assert_equal(i1.numpy(), i2.cpu().numpy())

    def test_spline_2d(self):
        if not torch.cuda.is_available():
            return

        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        input = torch.stack([input, input], dim=1)
        kernel_size = torch.LongTensor([5, 4])
        is_open_spline = torch.LongTensor([1, 0])

        a1, i1 = spline_cpu(input, kernel_size, is_open_spline, 1)
        a2, i2 = spline_gpu(input.cuda(),
                            kernel_size.cuda(), is_open_spline.cuda(), 1)

        assert_equal(a1.numpy(), a2.cpu().numpy())
        # assert_equal(i1.numpy(), i2.cpu().numpy())

    def test_spline_3d(self):
        if not torch.cuda.is_available():
            return

        input = torch.FloatTensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        input = torch.stack([input, input, input], dim=1)
        kernel_size = torch.LongTensor([5, 4, 4])
        is_open_spline = torch.LongTensor([1, 0, 0])
        a1, i1 = spline_cpu(input, kernel_size, is_open_spline, 1)

        a2, i2 = spline_gpu(input.cuda(),
                            kernel_size.cuda(), is_open_spline.cuda(), 1)

        # assert_equal(a1.numpy(), a2.cpu().numpy())
        # assert_equal(i1.numpy(), i2.cpu().numpy())
