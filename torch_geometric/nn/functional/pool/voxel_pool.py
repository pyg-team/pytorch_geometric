from torch_cluster import sparse_grid_cluster, dense_grid_cluster

from .pool import max_pool
from ....datasets.dataset import Data
from ...modules.spline_conv import repeat_to


def sparse_voxel_max_pool(data, size, start=None, transform=None):
    size = repeat_to(size, data.pos.size(1))
    size = data.pos.new(size)

    cluster, batch = sparse_grid_cluster(data.pos, size, data.batch, start)
    input, index, pos = max_pool(data.input, data.index, data.pos, cluster)

    data = Data(input, pos, index, None, data.target, batch)

    if transform is not None:
        data = transform(data)

    return data


def dense_voxel_max_pool(data, size, start=None, end=None, transform=None):
    size = repeat_to(size, data.pos.size(1))
    size = data.pos.new(size)

    cluster, C = dense_grid_cluster(data.pos, size, data.batch, start, end)
    input, index, pos = max_pool(data.input, data.index, data.pos, cluster, C)

    data = Data(input, pos, index, None, data.target)

    if transform is not None:
        data = transform(data)

    return data
