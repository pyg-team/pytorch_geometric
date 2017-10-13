from unittest import TestCase

from torch_geometric.nn.modules import SplineGCN


class SplineGCNTest(TestCase):
    def test_init(self):
        conv = SplineGCN(1, 10, dim=3, kernel_size=(4, 8), spline_degree=1)
        self.assertEqual(conv.in_features, 1)
        self.assertEqual(conv.out_features, 10)
        self.assertEqual(conv.kernel_size, (4, 8, 8))
        self.assertEqual(conv.spline_degree, (1, 1, 1))
        self.assertEqual(conv.weight.data.size(), (4, 8, 8, 1, 10))
        self.assertEqual(conv.bias.data.size(), (10, ))
        self.assertEqual(conv.__repr__(), 'SplineGCN(1, 10, '
                         'kernel_size=(4, 8, 8), spline_degree=(1, 1, 1))')

        conv = SplineGCN(
            1, 10, dim=2, kernel_size=4, spline_degree=(2, 1), bias=False)
        self.assertEqual(conv.in_features, 1)
        self.assertEqual(conv.out_features, 10)
        self.assertEqual(conv.kernel_size, (4, 4))
        self.assertEqual(conv.spline_degree, (2, 1))
        self.assertEqual(conv.weight.data.size(), (4, 4, 1, 10))
        self.assertIsNone(conv.bias)
        self.assertEqual(
            conv.__repr__(), 'SplineGCN(1, 10, '
            'kernel_size=(4, 4), spline_degree=(2, 1), bias=False)')

    def test_forward(self):
        pass
