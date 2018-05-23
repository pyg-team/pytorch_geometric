import numbers
from itertools import repeat

import torch
from torch_cluster import grid_cluster
from torch_geometric.data import Batch

from .consecutive import consecutive_cluster
from .pool import pool, max_pool


def voxel_grid(pos, size, start=None, end=None, batch=None, consecutive=True):
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    n, d = pos.size()

    if size is not None and isinstance(size, numbers.Number):
        size = list(repeat(size, d))
    size = pos.new(size) if size is not None else size

    if start is not None and isinstance(start, numbers.Number):
        start = list(repeat(start, d))
    start = pos.new(start) if start is not None else start

    if end is not None and isinstance(end, numbers.Number):
        end = list(repeat(end, d))
    end = pos.new(end) if end is not None else end

    if batch is not None:
        pos = torch.cat([pos, batch.unsqueeze(-1).type_as(pos)], dim=-1)
        size = torch.cat([size, size.new(1).fill_(1)], dim=-1)

        if start is not None:
            start = torch.cat([start, start.new(1).fill_(0)], dim=-1)
        if end is not None:
            end = torch.cat([end, end.new(1).fill_(batch.max() + 1)], dim=-1)

    cluster = grid_cluster(pos, size, start, end)

    if consecutive:
        cluster, batch = consecutive_cluster(cluster, batch)

    return cluster, batch


def sparse_voxel_grid_pool(data, size, start=None, end=None, transform=None):
    cluster, batch = voxel_grid(data.pos, size, start, end, data.batch, True)
    x = max_pool(data.x, cluster)
    edge_index, _, pos = pool(data.edge_index, cluster, None, data.pos)

    data = Batch(x=x, edge_index=edge_index, pos=pos, batch=batch)

    if transform is not None:
        data = transform(data)

    return data


def dense_voxel_grid_pool(data, size, start=None, end=None):
    cluster, _ = voxel_grid(data.pos, size, start, end, data.batch, False)
    x = max_pool(data.x, cluster)  # TODO: size
    return x
