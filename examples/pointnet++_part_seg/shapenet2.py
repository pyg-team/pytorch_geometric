import numpy as np
from torch_geometric.datasets import ShapeNet


class ShapeNet2(ShapeNet):
    r"""
    A simple wrapper of ShapeNet with:
        - Support sampling pointcloud to fixed number of points when
          npoints > 0.
        - Map label range from [1, C] to [0, C-1] for convinence.
    """

    def __init__(self,
                 root,
                 category,
                 npoints=0,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        super(ShapeNet2, self).__init__(root, category, train, transform,
                                        pre_transform, pre_filter)
        self.npoints = npoints

    def __getitem__(self, idx):
        data = super(ShapeNet2, self).__getitem__(idx)

        # Sample points
        if self.npoints > 0:
            choice = np.random.choice(
                data.pos.shape[0], self.npoints, replace=True)
            data.pos, data.y = data.pos[choice, :], data.y[choice]

        # Map label range from [1, C] to range [0, C-1]
        assert data.y.min().item() >= 1
        data.y = data.y - 1

        return data
