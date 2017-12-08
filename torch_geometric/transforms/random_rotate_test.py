from __future__ import division

from math import pi as PI
from unittest import TestCase
import torch

from ..datasets.data import Data
from .random_rotate import RandomRotate


class RandomRotateTest(TestCase):
    def test_random_rotate(self):
        position = [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
        position = torch.FloatTensor(position)
        data = Data(None, None, position, None)
        data = RandomRotate(PI / 2)(data)
