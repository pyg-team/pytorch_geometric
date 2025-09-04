from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import scatter


class SparseCrossEntropy(torch.autograd.Function):
    # We implement our own custom autograd function for this to avoid the
    # double gradient computation to `inputs`.
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        edge_label_index: Tensor,
        edge_label_weight: Optional[Tensor],
    ) -> Tensor:
        assert inputs.dim() == 2

        # Support for both positive and negative weights:
        # Positive weights scale the logits *after* softmax.
        # Negative weights scale the denominator *before* softmax:
        pos_y = edge_label_index
        neg_y = pos_weight = neg_weight = None

        if edge_label_weight is not None:
            pos_mask = edge_label_weight >= 0
            pos_y = edge_label_index[:, pos_mask]
            pos_weight = edge_label_weight[pos_mask]

            if pos_y.size(1) < edge_label_index.size(1):
                neg_mask = ~pos_mask
                neg_y = edge_label_index[:, neg_mask]
                neg_weight = edge_label_weight[neg_mask]

            if neg_y is not None and neg_weight is not None:
                inputs = inputs.clone()
                inputs[
                    neg_y[0],
                    neg_y[1],
                ] += neg_weight.abs().log().clamp(min=1e-12)

        logsumexp = inputs.logsumexp(dim=-1)
        ctx.save_for_backward(inputs, pos_y, pos_weight, logsumexp)

        out = inputs[pos_y[0], pos_y[1]]
        out.neg_().add_(logsumexp[pos_y[0]])
        if pos_weight is not None:
            out *= pos_weight

        return out.sum() / inputs.size(0)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[Tensor, None, None]:
        inputs, pos_y, pos_weight, logsumexp = ctx.saved_tensors

        grad_out = grad_out / inputs.size(0)
        grad_out = grad_out.expand(pos_y.size(1))

        if pos_weight is not None:
            grad_out = grad_out * pos_weight

        grad_logsumexp = scatter(grad_out, pos_y[0], dim=0,
                                 dim_size=inputs.size(0), reduce='sum')

        # Gradient computation of `logsumexp`: `grad * (self - result).exp()`
        grad_input = (inputs - logsumexp.view(-1, 1))
        grad_input.exp_()
        grad_input.mul_(grad_logsumexp.view(-1, 1))

        grad_input[pos_y[0], pos_y[1]] -= grad_out

        return grad_input, None, None


def sparse_cross_entropy(
    inputs: Tensor,
    edge_label_index: Tensor,
    edge_label_weight: Optional[Tensor] = None,
) -> Tensor:
    r"""A sparse-label variant of :func:`torch.nn.functional.cross_entropy`.
    In particular, the binary target matrix is solely given by sparse indices
    :obj:`edge_label_index`.

    Args:
        inputs (torch.Tensor): The predicted unnormalized logits of shape
            :obj:`[batch_size, num_classes]`.
        edge_label_index (torch.Tensor): The sparse ground-truth indices with
            shape :obj:`[2, num_labels]`.
        edge_label_weight (torch.Tensor, optional): The weight of ground-truth
            indices with shape :obj:`[num_labels]`. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`

    Example:
        >>> inputs = torch.randn(2, 3)
        >>> edge_label_index = torch.tensor([
        ...     [0, 0, 1],
        ...     [0, 1, 2],
        ... ])
        >>> loss = sparse_cross_entropy(inputs, edge_label_index)
        tensor(1.2919)
    """
    if edge_label_weight is not None:
        assert not edge_label_weight.requires_grad

    return SparseCrossEntropy.apply(
        inputs,
        edge_label_index,
        edge_label_weight,
    )
