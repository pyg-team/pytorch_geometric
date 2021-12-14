from torch_geometric.typing import Adj, Size

from torch import Tensor, ones
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LGConv(MessagePassing):
    r"""The Light Graph Convolution (LGC) operator from the `"LightGCN:
    Simplifying and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper. It supports the additional
    functionality of disabling normalization during neighborhood aggregation.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)}\sqrt{\deg(j)}}
        \mathbf{x}_j

    Args:
        aggr (string, optional): The aggregation scheme to use.
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be normalized using the symmetric normalization term:
            :math:`\frac{1}{\sqrt{\deg(i)}\sqrt{\deg(j)}}`.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    propagate_type = {'x': Tensor, 'norm': Tensor}

    def __init__(self, aggr: str = 'add', normalize: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        """"""
        norm = ones(x.size(0))
        if self.normalize:
            row, col = edge_index[0], edge_index[1]
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm, size=size)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j
