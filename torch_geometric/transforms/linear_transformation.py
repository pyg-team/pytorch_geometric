from typing import Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('linear_transformation')
class LinearTransformation(BaseTransform):
    r"""Transforms node positions :obj:`data.pos` with a square transformation
    matrix computed offline (functional name: :obj:`linear_transformation`)

    Args:
        matrix (Tensor): Tensor with shape :obj:`[D, D]` where :obj:`D`
            corresponds to the dimensionality of node positions.
    """
    def __init__(self, matrix: Tensor):
        if not isinstance(matrix, Tensor):
            matrix = torch.tensor(matrix)
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            f'Transformation matrix should be square (got {matrix.size()})')

        # Store the matrix as its transpose.
        # We do this to enable post-multiplication in `forward`.
        self.matrix = matrix.t()

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
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
