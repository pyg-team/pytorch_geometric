import torch
import scipy.spatial


class Delaunay(object):
    r"""Computes the delaunay triangulation of a set of points."""

    def __call__(self, data):
        print(data.pos.size())
        if data.pos.size(0) > 3:
            pos = data.pos.cpu().numpy()
            tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
            face = torch.from_numpy(tri.simplices)
        elif data.pos.size(0) == 3:
            face = torch.tensor([[0, 1, 2]])
        else:
            raise ValueError(
                'Not enough points to contruct Delaunay triangulation, got {} '
                'but expected at least 3'.format(data.pos.size(0)))

        data.face = face.t().contiguous().to(data.pos.device, torch.long)

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
