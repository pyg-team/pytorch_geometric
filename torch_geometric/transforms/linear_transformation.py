import torch


class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.

    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        # Store the matrix as its transpose. We do this to enable the 
        # postmultiplication in  __call__.
        self.matrix = matrix.transpose(0, 1)

    def __call__(self, data):
        pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos

        assert pos.size(-1) == self.matrix.size(-2), (
            'Node position matrix and transformation matrix have incompatible '
            'shape.')

        # We postmultiply the points by the transformation matrix of shape [D, D]
        # instead of premultiplying because data.pos has shape [N, D], and we want
        # to preserve this shape.
        data.pos = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())
