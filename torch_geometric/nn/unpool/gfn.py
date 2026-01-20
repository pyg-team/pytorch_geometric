import torch
from gfn import GFN


class GFNUnpooling(GFN):
    def __init__(self, in_size, out_graph, **kwargs):
        super().__init__(in_features=in_size, out_features=out_graph, **kwargs)

    def forward(self, x, pos_y, batch_x, batch_y):
        out = torch.empty((batch_y.shape[0]), *x.shape[1:],
                          dtype=self.weight.dtype, device=self.weight.device)
        for batch_label in batch_x.unique():
            mask = batch_y == batch_label
            pos = pos_y[mask, ...]
            x_ = x[batch_x == batch_label, ...]
            out[mask] = super().forward(x_.T, out_graph=pos).T
        return out
