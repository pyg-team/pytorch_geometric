import torch
from torch.nn.modules.instancenorm import _InstanceNorm
from torch_scatter import scatter_add
from torch_geometric.utils import degree


class InstanceNorm(_InstanceNorm):
    def __init__(self, channels, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm, self).__init__(channels, eps, momentum, affine,
                                           track_running_stats)

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1

        if self.training or not self.track_running_stats:
            count = degree(batch, batch_size, dtype=x.dtype).view(-1, 1)
            tmp = scatter_add(x, batch, dim=0, dim_size=batch_size)
            mean = tmp / count.clamp(min=1)

            tmp = (x - mean[batch])
            tmp = scatter_add(tmp * tmp, batch, dim=0, dim_size=batch_size)
            var = tmp / (count - 1).clamp(min=1)

        if self.training and self.track_running_stats:
            momentum = self.momentum
            self.running_mean = (
                1 - momentum) * self.running_mean + momentum * mean.mean(dim=0)
            self.running_var = (
                1 - momentum) * self.running_var + momentum * var.mean(dim=0)

        if self.track_running_stats:
            mean = self.running_mean.view(1, -1).expand(batch_size, -1)
            var = self.running_var.view(1, -1).expand(batch_size, -1)

        out = (x - mean[batch]) / torch.sqrt(var[batch] + self.eps)

        if self.affine:
            out = out * self.weight.view(1, -1) + self.bias.view(1, -1)

        return out
