from __future__ import division

import torch

from ..sparse import SparseTensor


class EuclideanAdj(object):
    def __call__(self, data):
        vertices, edges = data
        rows, cols = edges

        values = (vertices[cols] - vertices[rows])
        c = 1 / (2 * values.abs().max())
        values = c * values.float() + 0.5
        n, dim = vertices.size()
        adj = SparseTensor(edges, values, torch.Size([n, n, dim]))

        return vertices, adj
