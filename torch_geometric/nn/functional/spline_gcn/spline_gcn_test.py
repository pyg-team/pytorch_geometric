from unittest import TestCase
import torch
from torch.autograd import Variable
from numpy.testing import assert_almost_equal

from .spline_gcn import spline_gcn


class SplineGcnTest(TestCase):
    def test_forward_cpu(self):
        edges = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        values = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
        values = torch.FloatTensor(values)
        adj = torch.sparse.FloatTensor(edges, values, torch.Size([5, 5, 2]))

        kernel_size = torch.LongTensor([3, 4])
        is_open_spline = torch.LongTensor([1, 0])

        input = torch.FloatTensor([[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
        weight = torch.arange(0.5, 0.5 * 27, step=0.5).view(13, 2, 1)

        input, weight = Variable(input), Variable(weight)

        output = spline_gcn(adj, input, weight, kernel_size, is_open_spline, 1)

        expected_output = [
            [12.5 * 9 + 13 * 10 + 266],
            [12.5 * 1 + 13 * 2],
            [12.5 * 3 + 13 * 4],
            [12.5 * 5 + 13 * 6],
            [12.5 * 7 + 13 * 8],
        ]

        assert_almost_equal(output.cpu().data.numpy(), expected_output, 1)

    def test_forward_gpu(self):
        if not torch.cuda.is_available():
            return

        edges = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        values = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
        values = torch.FloatTensor(values)
        adj = torch.sparse.FloatTensor(edges, values, torch.Size([5, 5, 2]))

        kernel_size = torch.cuda.LongTensor([3, 4])
        is_open_spline = torch.cuda.LongTensor([1, 0])

        input = torch.FloatTensor([[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
        weight = torch.arange(0.5, 0.5 * 27, step=0.5).view(13, 2, 1)

        adj, input, weight = adj.cuda(), input.cuda(), weight.cuda()
        input, weight = Variable(input), Variable(weight)

        output = spline_gcn(adj, input, weight, kernel_size, is_open_spline, 1)

        expected_output = [
            [12.5 * 9 + 13 * 10 + 266],
            [12.5 * 1 + 13 * 2],
            [12.5 * 3 + 13 * 4],
            [12.5 * 5 + 13 * 6],
            [12.5 * 7 + 13 * 8],
        ]

        assert_almost_equal(output.cpu().data.numpy(), expected_output, 1)

    def test_backward_cpu(self):
        pass

    def test_backward_gpu(self):
        pass
