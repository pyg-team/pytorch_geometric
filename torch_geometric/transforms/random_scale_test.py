from unittest import TestCase
import torch

from ..datasets.data import Data
from .random_scale import RandomScale


class RandomScaleTest(TestCase):
    def test_random_scale(self):
        position = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
        position = torch.FloatTensor(position)
        data = Data(None, None, position, None)
        data = RandomScale(1)(data)
