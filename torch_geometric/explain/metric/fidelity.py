import torch
from torch import Tensor


def characterization_score(
    pos_fidelity: Tensor,
    neg_fidelity: Tensor,
    pos_weight: float = 0.5,
    neg_weight: float = 0.5,
) -> Tensor:
    r"""Returns the componentwise characterization score as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper:

    ..  math::
       \textrm{charact} = \frac{w_{+} + w_{-}}{\frac{w_{+}}{\textrm{fid}_{+}} +
        \frac{w_{-}}{1 - \textrm{fid}_{-}}}

    Args:
        pos_fidelity (torch.Tensor): The positive fidelity
            :math:`\textrm{fid}_{+}`.
        neg_fidelity (torch.Tensor): The negative fidelity
            :math:`\textrm{fid}_{-}`.
        pos_weight (float, optional): The weight :math:`w_{+}` for
            :math:`\textrm{fid}_{+}`. (default: :obj:`0.5`)
        neg_weight (float, optional): The weight :math:`w_{-}` for
            :math:`\textrm{fid}_{-}`. (default: :obj:`0.5`)
    """
    if (pos_weight + neg_weight) != 1.0:
        raise ValueError(f"The weights need to sum up to 1 "
                         f"(got {pos_weight} and {neg_weight})")

    denom = (pos_weight / pos_fidelity) + (neg_weight / (1. - neg_fidelity))
    return 1. / denom


def fidelity_curve_auc(
    pos_fidelity: Tensor,
    neg_fidelity: Tensor,
    x: Tensor,
) -> Tensor:
    r"""Returns the AUC for the fidelity curve as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    More precisely, returns the AUC of

    .. math::
        f(x) = \frac{\textrm{fid}_{+}}{1 - \textrm{fid}_{-}}

    Args:
        pos_fidelity (torch.Tensor): The positive fidelity
            :math:`\textrm{fid}_{+}`.
        neg_fidelity (torch.Tensor): The negative fidelity
            :math:`\textrm{fid}_{-}`.
        x (torch.Tensor): Tensor containing the points on the :math:`x`-axis.
            Needs to be sorted in ascending order.
    """
    if torch.any(neg_fidelity == 1):
        raise ValueError("There exists negative fidelity values containing 1, "
                         "leading to a division by zero")

    y = pos_fidelity / (1. - neg_fidelity)
    return auc(x, y)


def auc(x: Tensor, y: Tensor) -> Tensor:
    if torch.any(x.diff() < 0):
        raise ValueError("'x' must be given in ascending order")
    return torch.trapezoid(y, x)
