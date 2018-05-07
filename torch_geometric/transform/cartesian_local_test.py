import unittest
import torch

from ..datasets.dataset import Data
from .local_cartesian import LocalCartesian


class CartesianLocalTest(unittest.TestCase):
    def test_cartesian_local(self):
        pos = torch.FloatTensor([[-1, 0], [0, 0], [2, 0]])
        index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(None, pos, index, None, None)

        data = LocalCartesian()(data)
