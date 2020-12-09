from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class LabelPropagation(MessagePassing):
    r"""The label propagation operator from the `"Learning from Labeled and
    Unlabeled Datawith Label Propagation"
    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.

    Args:
        num_layers (int): The number of propagations.
        alpha (float, optional): The :math:`\alpha` coefficient.
            (default: :obj:`0.9`)
    """
    def __init__(self, num_layers: int, alpha: float = 0.9):
        super(LabelPropagation, self).__init__(aggr='add')
        self.num_layers = num_layers
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, y: Tensor, edge_index: Adj, mask: Tensor,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if mask.dtype == torch.long:
            idx = mask
            mask = mask.new_zeros(y.size(0), dtype=torch.bool)
            mask[idx] = True

        y = F.one_hot(y.view(-1)).to(torch.float)
        y[~mask] = 0.

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=False)

        res = (1 - self.alpha) * y
        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            y = self.propagate(edge_index, y=y, edge_weight=edge_weight,
                               size=None)
            y.mul_(self.alpha).add_(res).clamp_(0, 1)

        return y

    def message(self, y_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return y_j if edge_weight is None else edge_weight.view(-1, 1) * y_j

    def message_and_aggregate(self, adj_t: SparseTensor, y: Tensor) -> Tensor:
        return matmul(adj_t, y, reduce=self.aggr)

    def __repr__(self):
        return '{}(num_layers={}, alpha={})'.format(self.__class__.__name__,
                                                    self.num_layers,
                                                    self.alpha)
