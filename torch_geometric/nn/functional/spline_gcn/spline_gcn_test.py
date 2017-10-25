from unittest import TestCase
import torch
from torch.autograd import Variable
from numpy.testing import assert_almost_equal

from .spline_gcn import spline_gcn


class SplineGcnTest(TestCase):
    def test_forward(self):
        edges = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        values = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
        values = torch.FloatTensor(values)
        adj = torch.sparse.FloatTensor(edges, values, torch.Size([5, 5, 2]))

        kernel_size = torch.cuda.LongTensor([3, 4])
        is_open_spline = torch.cuda.LongTensor([1, 0])

        input = torch.FloatTensor([[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
        weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)

        adj, input, weight = adj.cuda(), input.cuda(), weight.cuda()
        input, weight = Variable(input), Variable(weight)

        output = spline_gcn(adj, input, weight, kernel_size, is_open_spline, 1)

        expected_output = [
            [2 * 9 + 2.5 * 10 + 266],
            [2 * 1 + 2.5 * 2],
            [2 * 3 + 2.5 * 4],
            [2 * 5 + 2.5 * 6],
            [2 * 7 + 2.5 * 8],
        ]

        assert_almost_equal(output.cpu().data.numpy(), expected_output, 1)

    def test_backward(self):
        pass
        # vertices = [[0, 0], [1, 1], [-2, 2], [-3, -3], [4, -4]]
        # vertices = torch.LongTensor(vertices)
        # edges = [[0, 0, 0, 0], [1, 2, 3, 4]]
        # edges = torch.LongTensor(edges)
        # adj = mesh_adj(vertices, edges, type=torch.DoubleTensor)
        # features = torch.randn(5, 2).double()
        # features = Variable(features, requires_grad=True)
        # weight = torch.randn(12, 2, 4).double()
        # weight = Variable(weight, requires_grad=True)

        # def op(features, weight):
        #     return spline_gcn(
        #         adj,
        #         features,
        #         weight,
        #         kernel_size=[3, 4],
        #         max_radius=sqrt(16 + 16),
        #         degree=1)

        # test = gradcheck(op, (features, weight), eps=1e-6, atol=1e-4)
        # self.assertTrue(test)
