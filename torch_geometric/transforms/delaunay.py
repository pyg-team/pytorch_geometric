from typing import List

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class _QhullTransform(BaseTransform):
    r"""Q-hull implementation of Delaunay triangulation, uses NumPy arrays."""
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        import scipy.spatial

        pos = data.pos.cpu().numpy()
        tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
        face = torch.from_numpy(tri.simplices)

        data.face = face.t().contiguous().to(data.pos.device, torch.long)
        return data


class _ShullTransform(BaseTransform):
    r"""Sweep-hull implementation of Delaunay triangulation, uses Torch
    tensors.
    """
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        from torch_delaunay.functional import shull2d

        face = shull2d(data.pos.cpu())
        data.face = face.t().contiguous().to(data.pos.device)
        return data


class _SequentialTransform(BaseTransform):
    r"""Runs the specifies transforms one after another sequentially.

    Returns the first successful result, and raises exception otherwise. All
    intermediate exceptions are suppressed except the last.
    """
    def __init__(self, transforms: List[BaseTransform]) -> None:
        if (num_transforms := len(transforms)) == 0:
            raise ValueError(
                f"at least one transform is expected, got {num_transforms}")
        self.transforms = transforms

    def forward(self, data: Data) -> Data:
        for i, transform in enumerate(self.transforms):
            try:
                return transform.forward(data)
            except ModuleNotFoundError as e:
                if i == len(self.transforms) - 1:
                    raise e
        return data


@functional_transform('delaunay')
class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points
    (functional name: :obj:`delaunay`).

    .. hint::
        Consider installing `torch_delaunay` python package to speed up
        computations of Delaunay tessellations.
    """
    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        if data.pos.size(0) < 2:
            data.edge_index = torch.tensor([], dtype=torch.long,
                                           device=data.pos.device).view(2, 0)
        if data.pos.size(0) == 2:
            data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long,
                                           device=data.pos.device)
        elif data.pos.size(0) == 3:
            data.face = torch.tensor([[0], [1], [2]], dtype=torch.long,
                                     device=data.pos.device)
        if data.pos.size(0) > 3:
            transform = _SequentialTransform([
                _QhullTransform(),
                _ShullTransform(),
            ])
            data = transform.forward(data)

        return data
