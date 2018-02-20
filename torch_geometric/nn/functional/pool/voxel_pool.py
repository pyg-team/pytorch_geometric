from torch_cluster import grid_cluster

from .pool import max_pool
from ....datasets.dataset import Data
from ...modules.spline_conv import repeat_to


def voxel_max_pool(data, size, offset=None, fake_nodes=False, transform=None):
    size = repeat_to(size, data.pos.size(1))
    size = data.pos.new(size)

    c, b = grid_cluster(data.pos, size, data.batch, offset, fake_nodes)
    b = None if fake_nodes else b
    input, index, pos = max_pool(data.input, data.index, data.pos, c)
    data = Data(input, pos, index, None, data.target, b)

    if transform is not None:
        data = transform(data)

    return data, c
