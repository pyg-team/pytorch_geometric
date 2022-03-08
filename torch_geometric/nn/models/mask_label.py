import torch


class MaskLabel(torch.nn.Module):
    r"""A label embedding layer that replicates the label masking from `"Masked
    Label Prediction: Unified Message Passing Model for Semi-Supervised
    Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

    In the forward pass both the labels, and a mask corresponding to which
    labels should be kept is provided. All entires that are not true in the
    mask are returned as zero.

    Args:
        num_classes (int): Size of the number of classes for the labels
        out_channels (int): Size of each output sample.
    """
    def __init__(self, num_classes, out_channels):
        super().__init__()

        self.emb = torch.nn.Embedding(num_classes, out_channels)
        self.out_channels = out_channels

    def forward(self, y: torch.Tensor, mask: torch.Tensor):
        out = torch.zeros(y.shape[0], self.out_channels, dtype=torch.float)
        out[mask] = self.emb(y[mask])
        return out

    @staticmethod
    def ratio_mask(mask: torch.Tensor, ratio: float, shuffle: bool = False):
        r"""Modifies the existing :obj:`mask` by additionally setting
        :obj:`ratio` of the :obj:`True` entries to :obj:`False`. Does not
        operate inplace.

        If shuffle is required the masking proportion is not exact.

        Args:
            mask (torch.Tensor): The mask to re-mask.
            ratio (float): The ratio of entries to remove.
            shuffle (bool): Whether or not to randomize which elements
                to change to :obj:`False`.
        """
        n = int(mask.sum())
        mask = torch.clone(mask)
        new_mask = torch.arange(1, n + 1) > ratio * n
        if shuffle:
            new_mask = new_mask[torch.randperm(n)]
        mask[mask == 1] = new_mask
        return mask
