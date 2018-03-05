from unittest import TestCase
import torch

from ..datasets.data import Data
from .random_translate import RandomTranslate


class RandomTranslateTest(TestCase):
    def test_random_translate(self):
        position = torch.FloatTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        data = Data(None, None, position, None)
        data = RandomTranslate(1)(data)
