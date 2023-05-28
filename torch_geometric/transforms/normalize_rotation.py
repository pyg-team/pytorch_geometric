import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('normalize_rotation')
class NormalizeRotation(BaseTransform):
    r"""Rotates all points according to the eigenvectors of the point cloud
    (functional name: :obj:`normalize_rotation`).
    If the data additionally holds normals saved in :obj:`data.normal`, these
    will be rotated accordingly.

    Args:
        max_points (int, optional): If set to a value greater than :obj:`0`,
            only a random number of :obj:`max_points` points are sampled and
            used to compute eigenvectors. (default: :obj:`-1`)
        sort (bool, optional): If set to :obj:`True`, will sort eigenvectors
            according to their eigenvalues. (default: :obj:`False`)
    """
    def __init__(self, max_points: int = -1, sort: bool = False):
        self.max_points = max_points
        self.sort = sort

    def forward(self, data: Data) -> Data:
        pos = data.pos

        if self.max_points > 0 and pos.size(0) > self.max_points:
            perm = torch.randperm(pos.size(0))
            pos = pos[perm[:self.max_points]]

        pos = pos - pos.mean(dim=0, keepdim=True)
        C = torch.matmul(pos.t(), pos)
        e, v = torch.linalg.eig(C)  # v[:,j] is j-th eigenvector
        e, v = torch.view_as_real(e), v.real

        if self.sort:
            indices = e[:, 0].argsort(descending=True)
            v = v.t()[indices].t()

        data.pos = torch.matmul(data.pos, v)

        if 'normal' in data:
            data.normal = F.normalize(torch.matmul(data.normal, v))

        return data
