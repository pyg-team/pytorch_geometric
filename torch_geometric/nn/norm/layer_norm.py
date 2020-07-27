import torch
from torch_scatter import scatter_add
from torch import nn
from torch_geometric.utils import degree
from typing import Optional


class LayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5, affine=False):
        super(LayerNorm, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(1, in_channels))
        self.bias = torch.nn.Parameter(torch.zeros(1, in_channels))
        self.affine = affine

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = int(batch.max().item() + 1)

        count = degree(batch, batch_size, dtype=x.dtype).view(-1, 1).clamp(min=1) * x.shape[1]
        tmp = scatter_add(x, batch, dim=0, dim_size=batch_size).sum(dim=1, keepdim=True)
        mean = tmp / count

        mean_diff = (x - mean[batch])
        tmp = scatter_add(mean_diff * mean_diff, batch, dim=0, dim_size=batch_size).sum(dim=1, keepdim=True)
        var = tmp / count

        if not self.affine:
            return mean_diff / (torch.sqrt(var[batch]) + self.eps)
        return (mean_diff / (torch.sqrt(var[batch]) + self.eps)) * self.weight + self.bias