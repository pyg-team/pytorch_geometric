import torch
from torch_geometric.utils import to_batch


def sort_pool(x, batch, k):
    x, _ = x.sort(dim=-1)

    fill_value = x.min().item() - 1
    batch_x, num_nodes = to_batch(x, batch, fill_value)
    B, N, D = batch_x.size()

    _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)

    batch_x = batch_x[:, :k].contiguous()
    batch_x[batch_x == fill_value] = 0
    x = batch_x.view(B, k * D)

    return x
