from unittest import TestCase
import torch

from .edgewise_spline_weighting_gpu import edgewise_spline_weighting_forward
from .spline import spline_weights


class EdgewiseSplineWeightingGPUTest(TestCase):
    def test_forward(self):
        col = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
        col = torch.FloatTensor(col)
        kernel_size = torch.LongTensor([3, 4])
        is_open_spline = torch.LongTensor([1, 0])

        amount, index = spline_weights(col, kernel_size, is_open_spline, 1)
        amount, index = amount.cuda(), index.cuda()

        input = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)
        input, weight = input.cuda(), weight.cuda()

        out = edgewise_spline_weighting_forward(input, weight, amount, index)
        print(out)
