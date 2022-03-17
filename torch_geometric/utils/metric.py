from typing import Optional

import torchmetrics.functional as tm_func
from deprecate import deprecated
from torch import Tensor


@deprecated(tm_func.accuracy, args_mapping={"pred": "preds"})
def accuracy(pred: Tensor, target: Tensor) -> float:
    r"""Computes the accuracy of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: float
    """


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


@deprecated(tm_func.jaccard_index, args_mapping={
    "pred": "preds",
    "batch": None,
    "omitnans": None
})
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
