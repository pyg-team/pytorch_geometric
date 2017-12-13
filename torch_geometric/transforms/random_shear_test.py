from unittest import TestCase
import torch

from ..datasets.data import Data
from .random_shear import RandomShear


class RandomShearTest(TestCase):
    def test_random_shear(self):
        position = [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
        position = torch.FloatTensor(position)
        data = Data(None, None, position, None)
        data = RandomShear(0.2)(data)
