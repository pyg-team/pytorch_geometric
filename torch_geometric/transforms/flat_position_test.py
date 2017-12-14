from unittest import TestCase
import torch
from numpy.testing import assert_equal

from ..datasets.data import Data
from .flat_position import FlatPosition


class FlatPositionTest(TestCase):
    def test_flat_position_without_input(self):
        position = torch.FloatTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        data = Data(None, None, position, None)

        data = FlatPosition()(data)

        expected_input = [[3], [4], [5]]
        expected_position = [[1, 2], [2, 3], [3, 4]]

        assert_equal(data.input.numpy(), expected_input)
        assert_equal(data.position.numpy(), expected_position)

    def test_flat_position_with_input(self):
        position = torch.FloatTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        input = torch.FloatTensor([[1, 0], [2, 1], [3, 2]])
        data = Data(input, None, position, None)

        data = FlatPosition()(data)

        expected_input = [[1, 0, 3], [2, 1, 4], [3, 2, 5]]
        expected_position = [[1, 2], [2, 3], [3, 4]]

        assert_equal(data.input.numpy(), expected_input)
        assert_equal(data.position.numpy(), expected_position)
