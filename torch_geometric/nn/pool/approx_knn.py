# This file has code to create a KNN graph with approximate
# nearest neighbors. We can call it approx-knn.

import torch
from torch import Tensor


def approx_nn(x: Tensor, y: Tensor, k: int, batch_x: Tensor = None,
              batch_y: Tensor = None) -> Tensor:
    from pynndescent import NNDescent

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    min_xy = min(x.min().item(), y.min().item())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max().item(), y.max().item())
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    # Default metric is euclidean
    # More can be found in the
    # documentation here - https://pynndescent.readthedocs.io/en/latest/
    index = NNDescent(x.detach().numpy())
    col, dist = index.query(y.detach().cpu(), k=k)
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = ~torch.isinf(dist).view(-1)
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return torch.stack([row, col], dim=0)


def approx_knn_graph(x: Tensor, k: int, batch: Tensor = None,
                     loop: bool = False,
                     flow: str = 'source_to_target') -> Tensor:
    assert flow in ['source_to_target', 'target_to_source']
    row, col = approx_nn(x, x, k if loop else k + 1, batch, batch)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)
