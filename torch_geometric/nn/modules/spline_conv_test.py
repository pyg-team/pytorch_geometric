import unittest

from .utils.repeat import repeat_to
from .spline_conv import SplineConv


class SplineConvTest(unittest.TestCase):
    def test_repeat_to(self):
        output = repeat_to(2, 5)
        self.assertEqual(output, [2, 2, 2, 2, 2])

        output = repeat_to([2, 4, 3], 5)
        self.assertEqual(output, [2, 4, 3, 3, 3])

        output = repeat_to([2, 4, 3, 7, 2], 5)
        self.assertEqual(output, [2, 4, 3, 7, 2])

    def test_init(self):
        conv = SplineConv(1, 10, dim=3, kernel_size=[4, 8], degree=2)
        self.assertEqual(conv.in_channels, 1)
        self.assertEqual(conv.out_channels, 10)
        self.assertEqual(conv.kernel_size.tolist(), [4, 8, 8])
        self.assertEqual(conv.is_open_spline.tolist(), [True, True, True])
        self.assertEqual(conv.degree, 2)
        self.assertEqual(conv.weight.data.size(), (4 * 8 * 8, 1, 10))
        self.assertEqual(conv.root_weight.data.size(), (1, 10))
        self.assertEqual(conv.bias.data.size(), (10, ))
