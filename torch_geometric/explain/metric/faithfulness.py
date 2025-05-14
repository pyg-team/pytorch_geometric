from typing import Optional

import torch
import torch.nn.functional as F

from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import MaskType, ModelMode, ModelReturnType


def unfaithfulness(
    explainer: Explainer,
    explanation: Explanation,
    top_k: Optional[int] = None,
) -> float:
    r"""Evaluates how faithful an :class:`~torch_geometric.explain.Explanation`
    is to an underlying GNN predictor, as described in the
    `"Evaluating Explainability for Graph Neural Networks"
    <https://arxiv.org/abs/2208.09339>`_ paper.

    In particular, the graph explanation unfaithfulness metric is defined as

    .. math::
        \textrm{GEF}(y, \hat{y}) = 1 - \exp(- \textrm{KL}(y || \hat{y}))

    where :math:`y` refers to the prediction probability vector obtained from
    the original graph, and :math:`\hat{y}` refers to the prediction
    probability vector obtained from the masked subgraph.
    Finally, the Kullback-Leibler (KL) divergence score quantifies the distance
    between the two probability distributions.

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
        top_k (int, optional): If set, will only keep the original values of
            the top-:math:`k` node features identified by an explanation.
            If set to :obj:`None`, will use :obj:`explanation.node_mask` as it
            is for masking node features. (default: :obj:`None`)
    """
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    if top_k is not None and explainer.node_mask_type == MaskType.object:
        raise ValueError("Cannot apply top-k feature selection based on a "
                         "node mask of type 'object'")

    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    x, edge_index = explanation.x, explanation.edge_index
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.get('prediction')
    if y is None:  # == ExplanationType.phenomenon
        y = explainer.get_prediction(x, edge_index, **kwargs)

    if node_mask is not None and top_k is not None:
        feat_importance = node_mask.sum(dim=0)
        _, top_k_index = feat_importance.topk(top_k)
        node_mask = torch.zeros_like(node_mask)
        node_mask[:, top_k_index] = 1.0

    y_hat = explainer.get_masked_prediction(x, edge_index, node_mask,
                                            edge_mask, **kwargs)

    if explanation.get('index') is not None:
        y, y_hat = y[explanation.index], y_hat[explanation.index]

    if explainer.model_config.return_type == ModelReturnType.raw:
        y, y_hat = y.softmax(dim=-1), y_hat.softmax(dim=-1)
    elif explainer.model_config.return_type == ModelReturnType.log_probs:
        y, y_hat = y.exp(), y_hat.exp()

    kl_div = F.kl_div(y.log(), y_hat, reduction='batchmean')
    return 1 - float(torch.exp(-kl_div))
