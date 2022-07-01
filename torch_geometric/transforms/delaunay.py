import scipy.spatial
import torch

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('delaunay')
class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points
    (functional name: :obj:`delaunay`)."""
    def __call__(self, data):
        if data.pos.size(1) < 2:
            data.edge_index = torch.tensor([], dtype=torch.long,
                                           device=data.pos.device).view(2, 0)
        if data.pos.size(1) == 2:
            data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long,
                                           device=data.pos.device)
        elif data.pos.size(1) == 3:
            data.face = torch.tensor([[0], [1], [2]], dtype=torch.long,
                                     device=data.pos.device)
        if data.pos.size(1) > 3:
            pos = data.pos.cpu().numpy()
            tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
            face = torch.from_numpy(tri.simplices)

            data.face = face.t().contiguous().to(data.pos.device, torch.long)

        return data
