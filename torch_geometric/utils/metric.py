from typing import Optional, Tuple

import torch
import torchmetrics.functional as tm_func
import torch.nn.functional as F
from deprecate import deprecated
from torch import Tensor
from torch_scatter import scatter_add


@deprecated(tm_func.accuracy, args_mapping={"pred": "preds"})
def accuracy(pred: Tensor, target: Tensor) -> float:
    r"""Computes the accuracy of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: float
    """


def true_positive(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return tm_func.confusion_matrix()


def true_negative(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out, device=pred.device)


def false_positive(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out, device=pred.device)


def false_negative(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out, device=pred.device)


@deprecated(tm_func.precision, args_mapping={"pred": "preds"})
def precision(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    r"""Computes the precision of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """


@deprecated(tm_func.recall, args_mapping={"pred": "preds"})
def recall(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    r"""Computes the recall of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """


@deprecated(tm_func.f1_score, args_mapping={"pred": "preds"})
def f1_score(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    r"""Computes the :math:`F_1` score of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """


def intersection_and_union(
        pred: Tensor, target: Tensor, num_classes: int,
        batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    r"""Computes intersection and union of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)
    """
    pred, target = F.one_hot(pred, num_classes), F.one_hot(target, num_classes)

    if batch is None:
        i = (pred & target).sum(dim=0)
        u = (pred | target).sum(dim=0)
    else:
        i = scatter_add(pred & target, batch, dim=0)
        u = scatter_add(pred | target, batch, dim=0)

    return i, u


@deprecated(tm_func.jaccard_index, args_mapping={"pred": "preds", "batch": None, "omitnans": None})
def mean_iou(pred: Tensor, target: Tensor, num_classes: int,
             batch: Optional[Tensor] = None, omitnans: bool = False) -> Tensor:
    r"""Computes the mean intersection over union score of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.
        omitnans (bool, optional): If set to :obj:`True`, will ignore any
            :obj:`NaN` values encountered during computation. Otherwise, will
            treat them as :obj:`1`. (default: :obj:`False`)

    :rtype: :class:`Tensor`
    """
