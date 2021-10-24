from typing import Union

from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class LinearTransformation(BaseTransform):
    r"""Transforms node positions :obj:`pos` with a square transformation
    matrix computed offline.

    Args:
        matrix (Tensor): tensor with shape :obj:`[D, D]` where :obj:`D`
            corresponds to the dimensionality of node positions.
    """
    def __init__(self, matrix: Tensor):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        # Store the matrix as its transpose.
        # We do this to enable post-multiplication in `__call__`.
        self.matrix = matrix.t()

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if not hasattr(store, 'pos'):
                continue

            pos = store.pos.view(-1, 1) if store.pos.dim() == 1 else store.pos
            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have '
                'incompatible shape')
            # We post-multiply the points by the transformation matrix instead
            # of pre-multiplying, because `pos` attribute has shape `[N, D]`,
            # and we want to preserve this shape.
            store.pos = pos @ self.matrix.to(pos.device, pos.dtype)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n{self.matrix.cpu().numpy()}\n)'
