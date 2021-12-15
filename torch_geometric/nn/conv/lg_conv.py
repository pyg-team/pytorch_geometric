from torch_geometric.typing import Adj

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


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
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be normalized using the symmetric normalization term:
            :math:`\frac{1}{\sqrt{\deg(i)}\sqrt{\deg(j)}}`.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    propagate_type = {'x': Tensor, 'norm': Tensor}

    def __init__(self, aggr: str = 'add', normalize: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        _, norm = gcn_norm(edge_index, num_nodes=x.size(0),
                           add_self_loops=False)
        return self.propagate(edge_index, x=x, norm=norm, size=None)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j
