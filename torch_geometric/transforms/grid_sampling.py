import re
from typing import List, Optional, Union

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import one_hot, scatter


@functional_transform('grid_sampling')
class GridSampling(BaseTransform):
    r"""Clusters points into fixed-sized voxels
    (functional name: :obj:`grid_sampling`).
    Each cluster returned is a new point based on the mean of all points
    inside the given cluster.

    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
    """
    def __init__(self, size: Union[float, List[float], Tensor],
                 start: Optional[Union[float, List[float], Tensor]] = None,
                 end: Optional[Union[float, List[float], Tensor]] = None):
        self.size = size
        self.start = start
        self.end = end

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes

        batch = data.get('batch', None)

        c = torch_geometric.nn.voxel_grid(data.pos, self.size, batch,
                                          self.start, self.end)
        c, perm = torch_geometric.nn.pool.consecutive.consecutive_cluster(c)

        for key, item in data:
            if bool(re.search('edge', key)):
                raise ValueError(
                    'GridSampling does not support coarsening of edges')

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == 'y':
                    item = scatter(one_hot(item), c, dim=0, reduce='sum')
                    data[key] = item.argmax(dim=-1)
                elif key == 'batch':
                    data[key] = item[perm]
                else:
                    data[key] = scatter(item, c, dim=0, reduce='mean')

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'
