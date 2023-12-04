from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


@functional_transform('target_indegree')
class TargetIndegree(BaseTransform):
    r"""Saves the globally normalized degree of target nodes
    (functional name: :obj:`target_indegree`).

    .. math::

        \mathbf{u}(i,j) = \frac{\deg(j)}{\max_{v \in \mathcal{V}} \deg(v)}

    in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)
    """
    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def forward(self, data: Data) -> Data:
        col, pseudo = data.edge_index[1], data.edge_attr

        deg = degree(col, data.num_nodes)

        if self.norm:
            deg = deg / (deg.max() if self.max is None else self.max)

        deg = deg[col]
        deg = deg.view(-1, 1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, deg.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = deg

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
