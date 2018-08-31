import torch
from torch_cluster import grid_cluster

from ..repeat import repeat


def voxel_grid(pos, batch, size, start=None, end=None):
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    num_nodes, dim = pos.size()

    size, start, end = repeat(size, dim), repeat(start, dim), repeat(end, dim)

    pos = torch.cat([pos, batch.unsqueeze(-1).type_as(pos)], dim=-1)
    size = size + [1]
    start = None if start is None else start + [0]
    end = None if end is None else end + [batch.max().item()]

    size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
    if start is not None:
        start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
    if end is not None:
        end = torch.tensor(end, dtype=pos.dtype, device=pos.device)

    return grid_cluster(pos, size, start, end)
