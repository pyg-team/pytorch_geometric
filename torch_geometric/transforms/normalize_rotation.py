import torch
import torch.nn.functional as F


class NormalizeRotation(object):
    r"""Rotates all points so that the eigenvectors overlie the axes of the
    Cartesian coordinate system.
    If the data additionally holds normals saved in :obj:`data.norm` these will
    be also rotated.

    Args:
        max_points (int, optional): If set to a value greater than :obj:`0`,
            only a random number of :obj:`max_points` points are sampled and
            used to compute eigenvectors. (default: :obj:`-1`)
    """

    def __init__(self, max_points=-1):
        self.max_points = max_points

    def __call__(self, data):
        pos = data.pos

        if self.max_points > 0 and pos.size(0) > self.max_points:
            perm = torch.randperm(pos.size(0))
            pos = pos[perm[:self.max_points]]

        pos = pos - pos.mean(dim=0, keepdim=True)
        C = torch.matmul(pos.t(), pos)
        e, v = torch.eig(C, eigenvectors=True)  # v[:,j] is j-th eigenvector

        data.pos = torch.matmul(data.pos, v)

        if 'norm' in data:
            data.norm = F.normalize(torch.matmul(data.norm, v))

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
