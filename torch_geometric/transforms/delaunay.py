import torch
import scipy.spatial


class Delaunay(object):
    r"""Computes the delaunay triangulation of a set of points."""

    def __call__(self, data):
        pos = data.pos.cpu().numpy()
        tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
        face = torch.from_numpy(tri.simplices).to(data.pos.device, torch.long)
        data.face = face.t().contiguous()
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
