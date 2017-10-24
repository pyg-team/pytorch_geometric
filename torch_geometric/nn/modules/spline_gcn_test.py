from unittest import TestCase

from .spline_gcn import _repeat_last_to_count, SplineGCN


class SplineGCNTest(TestCase):
    def test_repeat_last_to_count(self):
        output = _repeat_last_to_count(2, 5)
        self.assertEqual(output, [2, 2, 2, 2, 2])

        output = _repeat_last_to_count([2, 4, 3], 5)
        self.assertEqual(output, [2, 4, 3, 3, 3])

        output = _repeat_last_to_count([2, 4, 3, 7, 2], 5)
        self.assertEqual(output, [2, 4, 3, 7, 2])

    def test_init(self):
        conv = SplineGCN(1, 10, dim=3, kernel_size=[4, 8], degree=2)
        self.assertEqual(conv.in_features, 1)
        self.assertEqual(conv.out_features, 10)
        self.assertEqual(conv.kernel_size, [4, 8, 8])
        self.assertEqual(conv.is_open_spline, [True, True, True])
        self.assertEqual(conv.degree, 2)
        self.assertEqual(conv.weight.data.size(), (1 + 4 * 8 * 8, 1, 10))
        self.assertEqual(conv.bias.data.size(), (10, ))
        self.assertEqual(conv.__repr__(), 'SplineGCN(1, 10, '
                         'kernel_size=[4, 8, 8], is_open_spline=[True, True, '
                         'True], degree=2)')

        conv = SplineGCN(
            1, 10, dim=1, kernel_size=[4], is_open_spline=False, bias=False)
        self.assertEqual(conv.in_features, 1)
        self.assertEqual(conv.out_features, 10)
        self.assertEqual(conv.kernel_size, [4])
        self.assertEqual(conv.is_open_spline, [False])
        self.assertEqual(conv.degree, 1)
        self.assertEqual(conv.weight.data.size(), (1 + 4, 1, 10))
        self.assertIsNone(conv.bias)
        self.assertEqual(conv.__repr__(), 'SplineGCN(1, 10, kernel_size=[4], '
                         'is_open_spline=[False], degree=1, bias=False)')
