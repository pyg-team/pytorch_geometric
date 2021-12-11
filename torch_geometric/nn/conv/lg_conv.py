from torch_geometric.typing import Adj, Size

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class LGConv(MessagePassing):
    r"""The Light Graph Convolution (LGC) operator from the `"LightGCN:
    Simplifying and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

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
            :math:`\frac{1}{\sqrt{|\mathcal{N}_i|}\sqrt{|\mathcal{N}_j|}}`.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, aggr: str = "add", normalize: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        norm = None
        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm, size=size)

    def message(self, x_j, norm: Tensor = None):
        if norm is not None:
            return norm.view(-1, 1) * x_j

        return x_j
