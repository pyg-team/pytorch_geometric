import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.models import LabelPropagation
from torch_geometric.typing import Adj, OptTensor


class CorrectAndSmooth(torch.nn.Module):
    r"""The correct and smooth (C&S) post-processing model from the
    `"Combining Label Propagation And Simple Models Out-performs Graph Neural
    Networks"
    <https://arxiv.org/abs/2010.13993>`_ paper, where soft predictions
    :math:`\mathbf{Z}` (obtained from a simple base predictor) are
    first corrected based on ground-truth training
    label information :math:`\mathbf{Y}` and residual propagation

    .. math::
        \mathbf{e}^{(0)}_i &= \begin{cases}
            \mathbf{y}_i - \mathbf{z}_i, & \text{if }i
            \text{ is training node,}\\
            \mathbf{0}, & \text{else}
        \end{cases}

    .. math::
        \mathbf{E}^{(\ell)} &= \alpha_1 \mathbf{D}^{-1/2}\mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{E}^{(\ell - 1)} +
        (1 - \alpha_1) \mathbf{E}^{(\ell - 1)}

        \mathbf{\hat{Z}} &= \mathbf{Z} + \gamma \cdot \mathbf{E}^{(L_1)},

    where :math:`\gamma` denotes the scaling factor (either fixed or
    automatically determined), and then smoothed over the graph via label
    propagation

    .. math::
        \mathbf{\hat{z}}^{(0)}_i &= \begin{cases}
            \mathbf{y}_i, & \text{if }i\text{ is training node,}\\
            \mathbf{\hat{z}}_i, & \text{else}
        \end{cases}

    .. math::
        \mathbf{\hat{Z}}^{(\ell)} = \alpha_2 \mathbf{D}^{-1/2}\mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{\hat{Z}}^{(\ell - 1)} +
        (1 - \alpha_2) \mathbf{\hat{Z}}^{(\ell - 1)}

    to obtain the final prediction :math:`\mathbf{\hat{Z}}^{(L_2)}`.

    .. note::

        For an example of using the C&S model, see
        `examples/correct_and_smooth.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        correct_and_smooth.py>`_.

    Args:
        num_correction_layers (int): The number of propagations :math:`L_1`.
        correction_alpha (float): The :math:`\alpha_1` coefficient.
        num_smoothing_layers (int): The number of propagations :math:`L_2`.
        smoothing_alpha (float): The :math:`\alpha_2` coefficient.
        autoscale (bool, optional): If set to :obj:`True`, will automatically
            determine the scaling factor :math:`\gamma`. (default: :obj:`True`)
        scale (float, optional): The scaling factor :math:`\gamma`, in case
            :obj:`autoscale = False`. (default: :obj:`1.0`)
    """
    def __init__(self, num_correction_layers: int, correction_alpha: float,
                 num_smoothing_layers: int, smoothing_alpha: float,
                 autoscale: bool = True, scale: float = 1.0):
        super().__init__()
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers, correction_alpha)
        self.prop2 = LabelPropagation(num_smoothing_layers, smoothing_alpha)

    def forward(self, y_soft: Tensor, *args) -> Tensor:  # pragma: no cover
        r"""Applies both :meth:`correct` and :meth:`smooth`."""
        y_soft = self.correct(y_soft, *args)
        return self.smooth(y_soft, *args)

    def correct(self, y_soft: Tensor, y_true: Tensor, mask: Tensor,
                edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        r"""
        Args:
            y_soft (Tensor): The soft predictions :math:`\mathbf{Z}` obtained
                from a simple base predictor.
            y_true (Tensor): The ground-truth label information
                :math:`\mathbf{Y}` of training nodes.
            mask (LongTensor or BoolTensor): A mask or index tensor denoting
                which nodes were used for training.
            edge_index (Tensor or SparseTensor): The edge connectivity.
            edge_weight (Tensor, optional): The edge weights.
                (default: :obj:`None`)
        """

        numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
        assert y_true.size(0) == numel

        if y_true.dtype == torch.long and y_true.size(0) == y_true.numel():
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)

        error = torch.zeros_like(y_soft)
        error[mask] = y_true - y_soft[mask]

        if self.autoscale:
            smoothed_error = self.prop1(error, edge_index,
                                        edge_weight=edge_weight,
                                        post_step=lambda x: x.clamp_(-1., 1.))

            sigma = error[mask].abs().sum() / numel
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000)] = 1.0
            return y_soft + scale * smoothed_error
        else:

            def fix_input(x):
                x[mask] = error[mask]
                return x

            smoothed_error = self.prop1(error, edge_index,
                                        edge_weight=edge_weight,
                                        post_step=fix_input)
            return y_soft + self.scale * smoothed_error

    def smooth(self, y_soft: Tensor, y_true: Tensor, mask: Tensor,
               edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        r"""
        Args:
            y_soft (Tensor): The corrected predictions :math:`\mathbf{Z}`
                obtained from :meth:`correct`.
            y_true (Tensor): The ground-truth label information
                :math:`\mathbf{Y}` of training nodes.
            mask (LongTensor or BoolTensor): A mask or index tensor denoting
                which nodes were used for training.
            edge_index (Tensor or SparseTensor): The edge connectivity.
            edge_weight (Tensor, optional): The edge weights.
                (default: :obj:`None`)
        """
        numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
        assert y_true.size(0) == numel

        if y_true.dtype == torch.long and y_true.size(0) == y_true.numel():
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)

        y_soft = y_soft.clone()
        y_soft[mask] = y_true

        return self.prop2(y_soft, edge_index, edge_weight=edge_weight)

    def __repr__(self):
        L1, alpha1 = self.prop1.num_layers, self.prop1.alpha
        L2, alpha2 = self.prop2.num_layers, self.prop2.alpha
        return (f'{self.__class__.__name__}(\n'
                f'  correct: num_layers={L1}, alpha={alpha1}\n'
                f'  smooth:  num_layers={L2}, alpha={alpha2}\n'
                f'  autoscale={self.autoscale}, scale={self.scale}\n'
                ')')
