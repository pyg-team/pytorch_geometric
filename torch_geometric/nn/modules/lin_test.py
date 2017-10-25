from unittest import TestCase

from .lin import Lin


class LinTest(TestCase):
    def test_init(self):
        lin = Lin(1, 10)
        self.assertEqual(lin.in_features, 1)
        self.assertEqual(lin.out_features, 10)
        self.assertEqual(lin.weight.data.size(), (10, 1, 1))
        self.assertEqual(lin.bias.data.size(), (10, ))
        self.assertEqual(lin.__repr__(), 'Lin(1, 10)')

        lin = Lin(1, 10, bias=False)
        self.assertIsNone(lin.bias)
        self.assertEqual(lin.__repr__(), 'Lin(1, 10, bias=False)')
