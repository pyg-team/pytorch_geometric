from unittest import TestCase
import torch

# from numpy.testing import assert_equal

# from torch_geometric.nn.functional import gcn


class GraphConvTest(TestCase):
    def test_forward(self):
        a_2 = torch.FloatTensor([[1, 1, 0], [1, 1, 2], [0, 2, 1]])
        degree = torch.sum(a_2, dim=1)
        degree = degree.pow(-0.5)

        # c = torch.mm(degree.view(1, -1), a_2)
        pass

    def test_backward(self):
        pass
