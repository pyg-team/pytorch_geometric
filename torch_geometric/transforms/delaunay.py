from typing import List

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class _QhullTransform(BaseTransform):
    r"""Q-hull implementation of delaunay triangulation."""
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        import scipy.spatial

        pos = data.pos.cpu().numpy()
        tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
        face = torch.from_numpy(tri.simplices)

        data.face = face.t().contiguous().to(data.pos.device, torch.long)
        return data


class _ShullTransform(BaseTransform):
    r"""Sweep-hull implementation of delaunay triangulation."""
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        from torch_delaunay.functional import shull2d

        face = shull2d(data.pos.cpu())
        data.face = face.t().contiguous().to(data.pos.device)
        return data


class _SequentialTransform(BaseTransform):
    r"""Runs the first successful transformation.

    All intermediate exceptions are suppressed except the last.
    """
    def __init__(self, transforms: List[BaseTransform]) -> None:
        assert len(transforms) > 0
        self.transforms = transforms

    def forward(self, data: Data) -> Data:
        for i, transform in enumerate(self.transforms):
            try:
                return transform.forward(data)
            except ImportError as e:
                if i == len(self.transforms) - 1:
                    raise e
        return data


@functional_transform('delaunay')
class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points
    (functional name: :obj:`delaunay`).

    .. hint::
        Consider installing the
        `torch_delaunay <https://github.com/ybubnov/torch_delaunay>`_ package
        to speed up computation.
    """
    def __init__(self) -> None:
        self._transform = _SequentialTransform([
            _ShullTransform(),
            _QhullTransform(),
        ])

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        device = data.pos.device

        if data.pos.size(0) < 2:
            data.edge_index = torch.empty(2, 0, dtype=torch.long,
                                          device=device)
        elif data.pos.size(0) == 2:
            data.edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
        elif data.pos.size(0) == 3:
            data.face = torch.tensor([[0], [1], [2]], device=device)
        else:
            data = self._transform.forward(data)

        return data
