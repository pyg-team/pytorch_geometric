from unittest import TestCase
import torch
from numpy.testing import assert_equal

from .geodesic_error import geodesic_error, max_geodesic_error_accuracy


class GeodesiicErrorTest(TestCase):
    def test_geodesic_error(self):
        position = torch.LongTensor([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
        pred = torch.LongTensor([1, 0, 2])
        label = torch.LongTensor([0, 1, 2])
        output = geodesic_error(position, pred, label)

        expected_output = [1, 1, 0]

        assert_equal(output.numpy(), expected_output)

    def test_max_geodesic_error_accuracy(self):
        position = torch.LongTensor([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
        pred = torch.LongTensor([1, 0, 2])
        label = torch.LongTensor([0, 1, 2])
        output = max_geodesic_error_accuracy(position, pred, label, error=0.0)

        expected_output = 1

        self.assertEqual(output, expected_output)
