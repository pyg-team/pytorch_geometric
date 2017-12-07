from unittest import TestCase
import torch

from ..datasets.data import Data
from .random_scale import RandomScale


class RandomScaleTest(TestCase):
    def test_random_scale(self):
        position = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
        position = torch.FloatTensor(position) - 2
        data = Data(None, None, position, None)
        data = RandomScale(2)(data)
