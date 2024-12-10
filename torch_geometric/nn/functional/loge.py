import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import OptTensor


def _apply_loss_reduction(values: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return torch.mean(values)
    elif reduction == "sum":
        return torch.sum(values)
    elif reduction == "none":
        return values
    else:
        raise ValueError(
            f"Invalid reduction {reduction}, expected one of 'mean', 'sum', or"
            " 'none'.")


def loge(
    input: Tensor,
    target: Tensor,
    epsilon: float = 1 - math.log(2),
    weight: OptTensor = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the Log-:math:`\epsilon` loss between input logits and target.

    See :class:`~torch_geometric.nn.functional.LogELoss` for details.
    """
    ce = F.nll_loss(input, target, reduction="none")
    unweighted = torch.log(epsilon + ce) - math.log(epsilon)
    # Apply weights by target class
    weighted = (unweighted *
                weight[target] if weight is not None else unweighted)
    return _apply_loss_reduction(weighted, reduction)


def loge_with_logits(
    input: Tensor,
    target: Tensor,
    epsilon: float = 1 - math.log(2),
    weight: OptTensor = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the Log-:math:`\epsilon` loss between input logits and target.

    See :class:`~torch_geometric.nn.functional.LogEWithLogitsLoss` for details.
    """
    ce = F.cross_entropy(input, target, reduction="none")
    unweighted = torch.log(epsilon + ce) - math.log(epsilon)
    # Apply weights by target class
    weighted = (unweighted *
                weight[target] if weight is not None else unweighted)
    return _apply_loss_reduction(weighted, reduction)


def binary_loge(
    input: Tensor,
    target: Tensor,
    epsilon: float = 1 - math.log(2),
    weight: OptTensor = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the Log-:math:`\epsilon` loss between input probabilities and
    binary target.

    See :class:`~torch_geometric.nn.functional.BinaryLogELoss` for details.
    """
    ce = F.binary_cross_entropy(input, target, reduction="none")
    unweighted = torch.log(epsilon + ce) - math.log(epsilon)
    weighted = unweighted * weight if weight is not None else unweighted
    return _apply_loss_reduction(weighted, reduction)


def binary_loge_with_logits(
    input: Tensor,
    target: Tensor,
    epsilon: float = 1 - math.log(2),
    weight: OptTensor = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the Log-:math:`\epsilon` loss between input logits and binary
    target.

    See :class:`~torch_geometric.nn.functional.BinaryLogEWithLogitsLoss` for
    details.
    """
    ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    unweighted = torch.log(epsilon + ce) - math.log(epsilon)
    weighted = unweighted * weight if weight is not None else unweighted
    return _apply_loss_reduction(weighted, reduction)


class _LogELossBase(torch.nn.Module):
    def __init__(
        self,
        epsilon: float = 1 - math.log(2),
        weight: OptTensor = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction


class LogELoss(_LogELossBase):
    r"""The the :math:`\log\epsilon` loss from `"Bag of Tricks for Node
    Classification with Graph Neural Networks"
    <https://arxiv.org/abs/2103.13355>`_, which computes, before reduction:

    .. math::
        \log\left(\epsilon + \mathrm{CrossEntropy}(x, y)\right)
        - \log \epsilon,

    where :math:`\epsilon` is a constant.

    Uses similar syntax and semantics as :class:`~torch.nn.NLLLoss`.
    In particular, it is useful when training a classification problem with
    more than two classes, and the input is expected to contain probabilities.
    However this criterion is less sensitive to outliers, with a maximal
    gradient magnitude at decision boundaries, while still providing
    non-negligable signal for all misclassified examples.

    Args:
        epsilon (float): The constant :math:`\epsilon`, which adjusts the shape
            of the loss surface.  (default: :obj:`1-math.log(2)`)
        weight (Tensor, optional): aa manual rescaling weight given to each
            class. If given, it has to be a Tensor of size C. (default:
            :obj:`None`)
        reduction (str, optional): Specifies the reduction to apply to the
            output: :obj:`'none' | 'mean' | 'sum'`. :obj:`'none'`: no reduction
            will be applied, :obj:`'mean'`: the sum of the output will be
            divided by the number of elements in the output, :obj:`'sum'`: the
            output will be summed. (default: :obj:`'mean'`)
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """"""  # noqa: D419
        return loge(
            input=input,
            target=target,
            epsilon=self.epsilon,
            weight=self.weight,
            reduction=self.reduction,
        )


class LogEWithLogitsLoss(_LogELossBase):
    r"""The the :math:`\log\epsilon` loss from `"Bag of Tricks for Node
    Classification with Graph Neural Networks"
    <https://arxiv.org/abs/2103.13355>`_, which computes, before reduction:

    .. math::
        \log\left(\epsilon + \mathrm{CrossEntropy}(x, y)\right)
        - \log \epsilon,

    where :math:`\epsilon` is a constant.

    Uses similar syntax and semantics as :class:`~torch.nn.CrossEntropyLoss`.
    In particular, it is useful when training a classification problem with
    more than two classes, and the input is expected to contain unnormalized
    logits. However this criterion is less sensitive to outliers, with a
    maximal gradient magnitude at decision boundaries, while still providing
    non-negligable signal for all misclassified examples.

    Args:
        epsilon (float): The constant :math:`\epsilon`, which adjusts the shape
            of the loss surface.  (default: :obj:`1-math.log(2)`)
        weight (Tensor, optional): aa manual rescaling weight given to each
            class. If given, it has to be a Tensor of size C. (default:
            :obj:`None`)
        reduction (str, optional): Specifies the reduction to apply to the
            output: :obj:`'none' | 'mean' | 'sum'`. :obj:`'none'`: no reduction
            will be applied, :obj:`'mean'`: the sum of the output will be
            divided by the number of elements in the output, :obj:`'sum'`: the
            output will be summed. (default: :obj:`'mean'`)
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """"""  # noqa: D419
        return loge_with_logits(
            input=input,
            target=target,
            epsilon=self.epsilon,
            weight=self.weight,
            reduction=self.reduction,
        )


class BinaryLogELoss(_LogELossBase):
    r"""The the :math:`\log\epsilon` loss from `"Bag of Tricks for Node
    Classification with Graph Neural Networks"
    <https://arxiv.org/abs/2103.13355>`_, which computes, before reduction:

    .. math::
        \log\left(\epsilon + \mathrm{CrossEntropy}(x, y)\right) -
        \log \epsilon,

    where :math:`\epsilon` is a constant.

    Uses similar syntax and semantics as :class:`~torch.nn.BCELoss`.
    In particular, it is useful when training a binary classification problem,
    and the input is expected to contain probabilities.  However this criterion
    is less sensitive to outliers, with a maximal gradient magnitude at
    decision boundaries, while still providing non-negligable signal for all
    misclassified examples.

    Args:
        epsilon (float): The constant :math:`\epsilon`, which adjusts the shape
            of the loss surface.  (default: :obj:`1-math.log(2)`)
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size nbatch.
            (default: :obj:`None`)
        reduction (str, optional): Specifies the reduction to apply to the
            output: :obj:`'none' | 'mean' | 'sum'`. :obj:`'none'`: no reduction
            will be applied, :obj:`'mean'`: the sum of the output will be
            divided by the number of elements in the output, :obj:`'sum'`: the
            output will be summed. (default: :obj:`'mean'`)
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """"""  # noqa: D419
        return binary_loge(
            input=input,
            target=target,
            epsilon=self.epsilon,
            weight=self.weight,
            reduction=self.reduction,
        )


class BinaryLogEWithLogitsLoss(_LogELossBase):
    r"""The the :math:`\log\epsilon` loss from `"Bag of Tricks for Node
    Classification with Graph Neural Networks"
    <https://arxiv.org/abs/2103.13355>`_, which computes, before reduction:

    .. math::
        \log\left(\epsilon + \mathrm{CrossEntropy}(x, y)\right)
        - \log \epsilon,

    where :math:`\epsilon` is a constant.

    Uses similar syntax and semantics as :class:`~torch.nn.BCEWithLogitsLoss`.
    In particular, it is useful when training a binary classification problem,
    and the input is expected to contain unnormalized logits.  However this
    criterion is less sensitive to outliers, with a maximal gradient magnitude
    at decision boundaries, while still providing non-negligable signal for all
    misclassified examples.

    Args:
        epsilon (float): The constant :math:`\epsilon`, which adjusts the shape
            of the loss surface.  (default: :obj:`1-math.log(2)`)
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size nbatch.
            (default: :obj:`None`)
        reduction (str, optional): Specifies the reduction to apply to the
            output: :obj:`'none' | 'mean' | 'sum'`. :obj:`'none'`: no reduction
            will be applied, :obj:`'mean'`: the sum of the output will be
            divided by the number of elements in the output, :obj:`'sum'`: the
            output will be summed. (default: :obj:`'mean'`)
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """"""  # noqa: D419
        return binary_loge_with_logits(
            input=input,
            target=target,
            epsilon=self.epsilon,
            weight=self.weight,
            reduction=self.reduction,
        )
