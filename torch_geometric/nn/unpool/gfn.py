import torch


class GFNUnpooling(torch.nn.Module):
    def __init__(self, in_size, pos_x, **kwargs):
        try:
            import gfn  # noqa
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "GFNUnpooling requires `gfn` to be installed. "
                "Please install it via `pip install gfn-layer`.") from e
        super().__init__()
        self._gfn = gfn.GFN(in_features=in_size, out_features=pos_x, **kwargs)

    def forward(self, x, pos_y, batch_x, batch_y):
        out = torch.empty((batch_y.shape[0]), *x.shape[1:],
                          dtype=self._gfn.weight.dtype,
                          device=self._gfn.weight.device)
        for batch_label in batch_x.unique():
            mask = batch_y == batch_label
            pos = pos_y[mask, ...]
            x_ = x[batch_x == batch_label, ...]
            out[mask] = self._gfn(x_.T, out_graph=pos).T
        return out
