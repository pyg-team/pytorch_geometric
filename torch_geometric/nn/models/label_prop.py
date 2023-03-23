from typing import Callable, Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import one_hot, spmm


class LabelPropagation(MessagePassing):
    r"""The label propagation operator from the `"Learning from Labeled and
    Unlabeled Data with Label Propagation"
    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.

    .. note::

        For an example of using the :class:`LabelPropagation`, see
        `examples/label_prop.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        label_prop.py>`_.

    Args:
        num_layers (int): The number of propagations.
        alpha (float): The :math:`\alpha` coefficient.
    """
    def __init__(self, num_layers: int, alpha: float):
        super().__init__(aggr='add')
        self.num_layers = num_layers
        self.alpha = alpha

    @torch.no_grad()
    def forward(
        self,
        y: Tensor,
        edge_index: Adj,
        mask: OptTensor = None,
        edge_weight: OptTensor = None,
        post_step: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        r"""
        Args:
            y (torch.Tensor): The ground-truth label information
                :math:`\mathbf{Y}`.
            edge_index (torch.Tensor or SparseTensor): The edge connectivity.
            mask (torch.Tensor, optional): A mask or index tensor denoting
                which nodes are used for label propagation.
                (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The edge weights.
                (default: :obj:`None`)
            post_step (callable, optional): A post step function specified
                to apply after label propagation. If no post step function
                is specified, the output will be clamped between 0 and 1.
                (default: :obj:`None`)
        """
        if y.dtype == torch.long and y.size(0) == y.numel():
            y = one_hot(y.view(-1))

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=False)

        res = (1 - self.alpha) * out
        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)
            out.mul_(self.alpha).add_(res)
            if post_step is not None:
                out = post_step(out)
            else:
                out.clamp_(0., 1.)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_layers={self.num_layers}, '
                f'alpha={self.alpha})')
