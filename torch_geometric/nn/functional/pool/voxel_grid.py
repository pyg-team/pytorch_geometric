import torch
from torch_cluster import grid_cluster
from torch_geometric.data import Batch

from .consecutive import consecutive_cluster
from ...modules.utils.repeat import repeat_to
from .pool import pool, max_pool


def voxel_grid(pos, size, start=None, end=None, batch=None, consecutive=True):
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    n, d = pos.size()

    size = pos.new(repeat_to(size, d))
    start = pos.new(repeat_to(start, d)) if start is not None else start
    end = pos.new(repeat_to(end, d)) if end is not None else end

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
