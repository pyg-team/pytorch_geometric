from typing import Tuple

import torch
from torch import Tensor

from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import ExplanationType, ModelMode


def fidelity(
    explainer: Explainer,
    explanation: Explanation,
) -> Tuple[float, float]:
    r"""Evaluates the fidelity of an
    :class:`~torch_geometric.explain.Explainer` given an
    :class:`~torch_geometric.explain.Explanation`, as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    Fidelity evaluates the contribution of the produced explanatory subgraph
    to the initial prediction, either by giving only the subgraph to the model
    (fidelity-) or by removing it from the entire graph (fidelity+).
    The fidelity scores capture how good an explainable model reproduces the
    natural phenomenon or the GNN model logic.

    For **phenomenon** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = y_i) \|

        \textrm{fid}_{-} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_S} = y_i) \|

    For **model** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = \hat{y}_i)

        \textrm{fid}_{-} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_S} = \hat{y}_i)

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
    """
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.target
    if explainer.explanation_type == ExplanationType.phenomenon:
        y_hat = explainer.get_prediction(
            explanation.x,
            explanation.edge_index,
            **kwargs,
        )
        y_hat = explainer.get_target(y_hat)

    explain_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        node_mask,
        edge_mask,
        **kwargs,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        1. - node_mask if node_mask is not None else None,
        1. - edge_mask if edge_mask is not None else None,
        **kwargs,
    )
    complement_y_hat = explainer.get_target(complement_y_hat)

    if explanation.get('index') is not None:
        y = y[explanation.index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = y_hat[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    if explainer.explanation_type == ExplanationType.model:
        pos_fidelity = 1. - (complement_y_hat == y).float().mean()
        neg_fidelity = 1. - (explain_y_hat == y).float().mean()
    else:
        pos_fidelity = ((y_hat == y).float() -
                        (complement_y_hat == y).float()).abs().mean()
        neg_fidelity = ((y_hat == y).float() -
                        (explain_y_hat == y).float()).abs().mean()

    return float(pos_fidelity), float(neg_fidelity)


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
