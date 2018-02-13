from torch_cluster import grid_cluster

from .pool import max_pool, avg_pool
from ....datasets.dataset import Data
from ...modules.spline_conv import repeat_to


def voxel_max_pool(data, cell_size, transform=None):
    cell_size = repeat_to(cell_size, data.pos.size(1))
    cell_size = data.pos.new(cell_size)

    cluster, batch = grid_cluster(data.pos, cell_size, data.batch)
    input, index, pos = max_pool(data.input, data.index, data.pos, cluster)

    data = Data(input, pos, index, None, data.target, batch)

    if transform is not None:
        data = transform(data)

    return data, cluster


def voxel_avg_pool(data, cell_size, transform=None):
    cell_size = repeat_to(cell_size, data.pos.size(1))
    cell_size = data.pos.new(cell_size)

    cluster, batch = grid_cluster(data.pos, cell_size, data.batch)
    input, index, pos = avg_pool(data.input, data.index, data.pos, cluster)

    data = Data(input, pos, index, None, data.target, batch)

    if transform is not None:
        data = transform(data)

    return data, cluster
