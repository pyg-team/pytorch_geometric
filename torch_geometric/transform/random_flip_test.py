import unittest
import torch

from ..datasets.dataset import Data
from .random_flip import flip


class CartesianLocalTest(unittest.TestCase):
    def test_flip_x(self):
        pos = torch.FloatTensor([[-1, 3], [0, 0], [2, -2]])
        data = Data(None, pos, None, None, None)

        data = flip(data, axis=0)
        self.assertEqual(data.pos.tolist(), [[2, 3], [1, 0], [-1, -2]])

    def test_flip_y(self):
        pos = torch.FloatTensor([[-1, 3], [0, 0], [2, -2]])
        data = Data(None, pos, None, None, None)

        data = flip(data, axis=1)
        self.assertEqual(data.pos.tolist(), [[-1, -2], [0, 1], [2, 3]])
