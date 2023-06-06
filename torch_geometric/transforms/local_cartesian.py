import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('local_cartesian')
class LocalCartesian(BaseTransform):
    r"""Saves the relative Cartesian coordinates of linked nodes in its edge
    attributes (functional name: :obj:`local_cartesian`). Each coordinate gets
    *neighborhood-normalized* to the specified interval.

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized. (default: :obj:`True`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        norm_range (tuple, optional): Tuple specifying the range for normalization.
            Each element of the tuple represents the lower and upper bounds for a
            coordinate dimension. (default: :obj:`(0, 1)`)
    """
    def __init__(self, norm=True, cat=True, norm_range=(0, 1)):
        self.norm = norm
        self.cat = cat
        self.norm_range = norm_range

    def forward(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = torch.empty(row.size(0), pos.size(1), device=pos.device)
        torch.sub(pos[row], pos[col], out=cart)  # In-place subtraction

        max_value = torch.empty(pos.size(0), device=pos.device)
        max_value.fill_(float('-inf'))

        for i in range(row.size(0)):
            cart_abs = cart[i].abs()
            max_value.index_copy_(0, col,
                                  torch.maximum(max_value[col], cart_abs))

        max_value = torch.max(max_value)

        if self.norm:
            norm_range_min, norm_range_max = self.norm_range
            norm_factor = 2 * max_value
            norm_factor[norm_factor == 0] = 1  # Avoid division by zero
            cart.div_(norm_factor).mul_(norm_range_max - norm_range_min).add_(
                norm_range_min)  # In-place normalization
        else:
            cart = cart

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data
