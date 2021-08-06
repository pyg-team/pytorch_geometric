import torch

from torch_geometric.transforms import BaseTransform


class LinearTransformation(BaseTransform):
    r"""Transforms node positions with a square transformation matrix computed
    offline.

    Args:
        matrix (Tensor): tensor with shape :obj:`[D, D]` where :obj:`D`
            corresponds to the dimensionality of node positions.
    """
    def __init__(self, matrix):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        # Store the matrix as its transpose.
        # We do this to enable post-multiplication in `__call__`.
        self.matrix = matrix.t()

    def __call__(self, data):
        pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos

        assert pos.size(-1) == self.matrix.size(-2), (
            'Node position matrix and transformation matrix have incompatible '
            'shape.')

        # We post-multiply the points by the transformation matrix instead of
        # pre-multiplying, because `data.pos` has shape `[N, D]`, and we want
        # to preserve this shape.
        data.pos = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())
