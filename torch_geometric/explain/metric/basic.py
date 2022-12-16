from typing import List, Optional, Tuple, Union

from torch import Tensor

METRICS = ['accuracy', 'recall', 'precision', 'f1_score', 'auroc']


def groundtruth_metrics(
    pred_mask: Tensor,
    target_mask: Tensor,
    metrics: Optional[Union[str, List[str]]] = None,
    threshold: float = 0.5,
) -> Union[float, Tuple[float, ...]]:
    r"""Compares and evaluates an explanation mask with the ground-truth
    explanation mask.

    Args:
        pred_mask (torch.Tensor): The prediction mask to evaluate.
        target_mask (torch.Tensor): The ground-truth target mask.
        metrics (str or List[str], optional). The metrics to return
            (:obj:`"accuracy"`, :obj:`"recall"`, :obj:`"precision"`,
            :obj:`"f1_score"`, :obj:`"auroc"`). (default: :obj:`["accuracy",
            "recall", "precision", "f1_score", "auroc"]`)
        threshold (float, optional): The threshold value to perform hard
            thresholding of :obj:`mask` and :obj:`groundtruth`.
            (default: :obj:`0.5`)
    """
    import torchmetrics

    if metrics is None:
        metrics = METRICS

    if isinstance(metrics, str):
        metrics = [metrics]

    if not isinstance(metrics, (tuple, list)):
        raise ValueError(f"Expected metrics to be a string or a list of "
                         f"strings (got {type(metrics)})")

    pred_mask = pred_mask.view(-1)
    target_mask = (target_mask >= threshold).view(-1)

    outs = []
    for metric in metrics:
        if metric not in METRICS:
            raise ValueError(f"Encountered invalid metric {metric}")

        fn = getattr(torchmetrics.functional, metric)
        if metric in {'auroc'}:
            out = fn(pred_mask, target_mask, 'binary')
        else:
            out = fn(pred_mask, target_mask, 'binary', threshold)

        outs.append(float(out))

    return tuple(outs) if len(outs) > 1 else outs[0]
