import re

import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add


class GridSampling(object):
    r"""Samples points depending on :obj:`subsampling_param`.

    Args:
        subsampling_param (int): The subsampling parameters used to map the pointcloud on a grid.
        num_classes (int): If the data contains labels, within key `y`, then it will be used to create label grid pooling. 
    """

    def __init__(self, subsampling_param, num_classes):
        self._subsampling_param = subsampling_param
        self._num_classes = num_classes

    def __call__(self, data):
        num_nodes = data.num_nodes

        pos = data.pos
        batch = data.batch

        pool = voxel_grid(pos, batch, self._subsampling_param)
        pool, _ = consecutive_cluster(pool)

        for key, item in data:
            if bool(re.search('edge', key)):
                continue

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == 'y':
                    one_hot = torch.zeros((item.shape[0], self._num_classes))\
                        .scatter(1, item.unsqueeze(-1), 1)

                    aggr_labels = scatter_add(one_hot, pool, dim=0)
                    data[key] = torch.argmax(aggr_labels, -1)
                else:
                    data[key] = pool_pos(pool, item).to(item.dtype)
        return data

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._subsampling_param, self._num_classes)