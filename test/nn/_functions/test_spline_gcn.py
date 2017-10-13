from unittest import TestCase

import torch


class SplineGcnTest(TestCase):
    def test_forward(self):
        print('drin')
        i = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        v1 = torch.FloatTensor([1, 2, 3, 4, 5, 6])
        v2 = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                                [11, 12]])
        a = torch.sparse.FloatTensor(i, v2, torch.Size([3, 3, 2]))
