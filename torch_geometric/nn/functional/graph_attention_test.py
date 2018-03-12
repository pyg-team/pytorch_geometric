from math import exp as e
from unittest import TestCase

import torch
from torch.autograd import Variable as Var

from .graph_attention import graph_attention


class GraphAttentionTest(TestCase):
    def test_attention(self):
        x = [[1, 4], [2, 3], [3, 2], [4, 1]]
        index = [[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 2, 0, 1, 0]]
        weight = [[0.1, 0.9], [0.2, 0.8]]
        att_weight = [[0.1, 0.2, 0.3, 0.4]]
        x, index = torch.Tensor(x), torch.LongTensor(index)
        weight, att_weight = torch.Tensor(weight), torch.Tensor(att_weight)
        x, weight, att_weight = Var(x), Var(weight), Var(att_weight)

        output = graph_attention(x, index, True, 0.2, 0, weight, att_weight)

        W = [[0.9, 4.1], [0.8, 4.2], [0.7, 4.3], [0.6, 4.4]]
        # W_cat = [
        #     [0.9, 4.1, 0.8, 4.2],
        #     [0.9, 4.1, 0.7, 4.3],
        #     [0.9, 4.1, 0.6, 4.4],
        #     [0.8, 4.2, 0.9, 4.1],
        #     [0.8, 4.2, 0.7, 4.3],
        #     [0.7, 4.3, 0.9, 4.1],
        #     [0.7, 4.3, 0.8, 4.2],
        #     [0.6, 4.4, 0.9, 4.1],
        #     [0.9, 4.1, 0.9, 4.1],
        #     [0.8, 4.2, 0.8, 4.2],
        #     [0.7, 4.3, 0.7, 4.3],
        #     [0.6, 4.4, 0.6, 4.4],
        # ]
        # W_cat_dot_att_weight = [
        #     2.83, 2.84, 2.85, 2.83, 2.85, 2.84, 2.85, 2.85, 2.82, 2.84, 2.86,
        #     2.88
        # ]
        a = [
            e(2.83) / (e(2.83) + e(2.84) + e(2.85) + e(2.82)),
            e(2.84) / (e(2.83) + e(2.84) + e(2.85) + e(2.82)),
            e(2.85) / (e(2.83) + e(2.84) + e(2.85) + e(2.82)),
            e(2.83) / (e(2.83) + e(2.85) + e(2.84)),
            e(2.85) / (e(2.83) + e(2.85) + e(2.84)),
            e(2.84) / (e(2.84) + e(2.85) + e(2.86)),
            e(2.85) / (e(2.84) + e(2.85) + e(2.86)),
            e(2.85) / (e(2.85) + e(2.88)),
            e(2.82) / (e(2.83) + e(2.84) + e(2.85) + e(2.82)),
            e(2.84) / (e(2.83) + e(2.85) + e(2.84)),
            e(2.86) / (e(2.84) + e(2.85) + e(2.86)),
            e(2.88) / (e(2.85) + e(2.88)),
        ]

        expected = [
            [
                a[0] * W[1][0] + a[1] * W[2][0] + a[2] * W[3][0] +
                a[8] * W[0][0],
                a[0] * W[1][1] + a[1] * W[2][1] + a[2] * W[3][1] +
                a[8] * W[0][1],
            ],
            [
                a[3] * W[0][0] + a[4] * W[2][0] + a[9] * W[1][0],
                a[3] * W[0][1] + a[4] * W[2][1] + a[9] * W[1][1],
            ],
            [
                a[5] * W[0][0] + a[6] * W[1][0] + a[10] * W[2][0],
                a[5] * W[0][1] + a[6] * W[1][1] + a[10] * W[2][1],
            ],
            [
                a[7] * W[0][0] + a[11] * W[3][0],
                a[7] * W[0][1] + a[11] * W[3][1],
            ],
        ]
        print(output)

    def test_multi_head_attention(self):
        x = [[1, 4], [2, 3], [3, 2], [4, 1]]
        index = [[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 2, 0, 1, 0]]
        weight = [[[0.1, 0.9], [0.1, 0.9]], [[0.2, 0.8], [0.2, 0.8]]]
        att_weight = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
        x, index = torch.Tensor(x), torch.LongTensor(index)
        weight, att_weight = torch.Tensor(weight), torch.Tensor(att_weight)
        x, weight, att_weight = Var(x), Var(weight), Var(att_weight)

        output = graph_attention(x, index, True, 0.2, 0, weight, att_weight)
        print(output)
