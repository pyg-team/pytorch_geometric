import torch
from torch import Tensor


class MaskLabel(torch.nn.Module):
    r"""The label embedding and masking layer from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    Here, node labels :obj:`y` are merged to the initial node features :obj:`x`
    for a subset of their nodes according to :obj:`mask`.

    .. note::

        For an example of using :class:`MaskLabel`, see
        `examples/unimp_arxiv.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/unimp_arxiv.py>`_.


    Args:
        num_classes (int): The number of classes.
        out_channels (int): Size of each output sample.
        method (str, optional): If set to :obj:`"add"`, label embeddings are
            added to the input. If set to :obj:`"concat"`, label embeddings are
            concatenated. In case :obj:`method="add"`, then :obj:`out_channels`
            needs to be identical to the input dimensionality of node features.
            (default: :obj:`"add"`)
    """
    def __init__(self, num_classes: int, out_channels: int,
                 method: str = "add"):
        super().__init__()

        self.method = method
        if method not in ["add", "concat"]:
            raise ValueError(
                f"'method' must be either 'add' or 'concat' (got '{method}')")

        self.emb = torch.nn.Embedding(num_classes, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """"""
        if self.method == "concat":
            out = x.new_zeros(y.size(0), self.emb.weight.size(-1))
            out[mask] = self.emb(y[mask])
            return torch.cat([x, out], dim=-1)
        else:
            x = torch.clone(x)
            x[mask] += self.emb(y[mask])
            return x

    @staticmethod
    def ratio_mask(mask: torch.Tensor, ratio: float, shuffle: bool = False):
        r"""Modifies :obj:`mask` by setting :obj:`ratio` of :obj:`True`
        entries to :obj:`False`. Does not operate in-place.

        Args:
            mask (torch.Tensor): The mask to re-mask.
            ratio (float): The ratio of entries to remove.
            shuffle (bool, optional): Whether to randomize which elements to
                change to :obj:`False`. (default: :obj:`False`)
        """
        n = int(mask.sum())
        mask = torch.clone(mask)
        new_mask = torch.arange(1, n + 1) > ratio * n
        if shuffle:
            new_mask = new_mask[torch.randperm(n)]
        mask[mask == 1] = new_mask
        return mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
