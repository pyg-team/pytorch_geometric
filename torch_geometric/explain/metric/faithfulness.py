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
    is to an underyling GNN predictor, as described in the
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




# Probing GNN Explainers: A Rigorous Theoretical and Empirical Analysis of GNN Explanation Methods

# code for rewiring edges, should be an input param rewiring_edges = True False in faitfulness function
def rewire_edges(x, edge_index, degree):
    # Convert to networkx graph for rewiring edges
    data = Data(x=x, edge_index=edge_index)
    G = convert.to_networkx(data, to_undirected=True)
    rewired_G = swap(G, nswap=degree, max_tries=degree * 25, seed=912)
    rewired_adj_mat = adj_mat(rewired_G)
    rewired_edge_indexes = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
    return rewired_edge_indexes


def faithfulness(explainer: Explainer, explanation: Explanation, threshold_config: ThresholdConfig = None,
                 num_iterations=10, rewirings_edges=True,
                 **kwargs):
    r"""Evaluates how faithful an :class:`~torch_geometric.explain.Explanation`
       is to an underyling GNN predictor, as described in the
       `"Probing GNN Explainers: A Rigorous Theoretical and Empirical Analysis of GNN Explanation Methods"
       <https://arxiv.org/abs/2106.09078>`_ paper.
    """
    # TODO: all this to a config or to params
    degree = 3
    pert_loc = 0
    pert_scale = 0.0001

    # getting subgraph
    subgraph = explanation.get_explanation_subgraph()
    if threshold_config is not None:
        subgraph = subgraph.threshold(threshold_config)

    node_mask, edge_mask = subgraph.node_mask, subgraph.edge_mask

    # getting perturbated_nodes
    # TODO : add a function with diffent option of perturbasion uniform,gaussian etc.
    sub_x = subgraph.x
    perturbed_nodes = [sub_x.clone()]
    for i in range(num_iterations):
        cont_noise = np.random.normal(loc=pert_loc, scale=pert_scale, size=sub_x.shape)
    sub_x += cont_noise
    perturbed_nodes.append(sub_x.clone().float())
    sub_x -= cont_noise

    # TODO: need to be done properly in explanation by using mapping it is just to test
    sub_index = torch.arange(explanation.index.shape[0])

    output_diff_norm = 0
    for i in range(num_iterations):
        rewired_edge_index = subgraph.edge_index
        if rewirings_edges and i != 0:
            try:
                rewired_edge_index = rewire_edges(x=perturbed_nodes[i],
                                                  edge_index=subgraph.edge_index,
                                                  degree=degree)
            except:
                # TODO: warning
                continue

            # predictions for perturbated features
            org_vec = explainer.get_prediction(
                perturbed_nodes[i],
                rewired_edge_index
            )
            org_softmax = torch.softmax(org_vec, dim=-1)

            # predictions for masked node features using explanations
            exp_vec = explainer.get_masked_prediction(
                perturbed_nodes[i],
                rewired_edge_index,
                node_mask,
                edge_mask
            )
            exp_softmax = torch.softmax(exp_vec, dim=-1)

            if explanation.index is not None:
                org_softmax = org_softmax[sub_index]
            exp_softmax = exp_softmax[sub_index]

            output_diff_norm += torch.norm(exp_softmax - org_softmax).item()
            # print(i, output_diff_norm)

    return output_diff_norm / num_iterations