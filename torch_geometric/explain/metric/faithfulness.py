from typing import Tuple

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.metrics.utils import perturb_node_features

# TODO: write function descriptions etc.
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


def faithfulness(explainer: Explainer, explanation: Explanation,
                 threshold_config: ThresholdConfig = None, num_iterations=10,
                 rewirings_edges=True, **kwargs):

    # TODO: all this to a config or to params
    degree = 3
    pert_loc = 0
    pert_scale = 0.0001

    #getting subgraph
    subgraph = explanation.get_explanation_subgraph()
    if threshold_config is not None:
        subgraph = subgraph.threshold(threshold_config)

    node_mask, edge_mask = subgraph.node_mask, subgraph.edge_mask

    # getting perturbated_nodes
    # TODO : add a function with diffent option of perturbasion uniform,gaussian etc.
    sub_x = subgraph.x
    perturbed_nodes = [sub_x.clone()]
    for i in range(num_iterations):
        cont_noise = np.random.normal(loc=pert_loc, scale=pert_scale,
                                      size=sub_x.shape)
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
                rewired_edge_index = rewire_edges(
                    x=perturbed_nodes[i], edge_index=subgraph.edge_index,
                    degree=degree)
            except:
                #TODO: warning
                continue

        # predictions for perturbated features
        org_vec = explainer.get_prediction(perturbed_nodes[i],
                                           rewired_edge_index)
        org_softmax = torch.softmax(org_vec, dim=-1)

        # predictions for masked node features using explanations
        exp_vec = explainer.get_masked_prediction(perturbed_nodes[i],
                                                  rewired_edge_index,
                                                  node_mask, edge_mask)
        exp_softmax = torch.softmax(exp_vec, dim=-1)

        if explanation.index is not None:
            org_softmax = org_softmax[sub_index]
            exp_softmax = exp_softmax[sub_index]

        output_diff_norm += torch.norm(exp_softmax - org_softmax).item()
        # print(i, output_diff_norm)

    return output_diff_norm / num_iterations


# This is from Evaluating Explainability for Graph Neural Networks
# In particular, we obtain the prediction probability vector  ŷu  using the GNN, i.e.  ŷu=f(Su) , and using the explanation, i.e.  ŷu′=f(Su′) ,
# where we generate a masked subgraph  Su′  by only keeping the original values of the top-k features identified by an explanation, and get their
# respective predictions  ŷu′ . Compute faithfulness by using KL divergence.
def faithfulness_node(explainer: Explainer, explanation: Explanation,
                      threshold_config: ThresholdConfig = None, **kwargs):
    # getting original predictions for index

    out_vec = explanation.prediction[explanation.index]
    out_softmax = F.softmax(out_vec, dim=-1)
    # takes an explanation mask and applies it to any given node and its subgraph to generate a new masked subgraph, which is then passed as input to the model f to obtain a prediction.
    sub_index = (explanation.node_mask > 0).cumsum(
        dim=0)[explanation.index] - 1
    subgraph = explanation.get_explanation_subgraph()

    if threshold_config is not None:
        topk_explanation = subgraph.threshold(threshold_config)
        subgraph = topk_explanation.get_explanation_subgraph()

    with torch.no_grad():
        pert_vec = explainer.get_prediction(subgraph.x, subgraph.edge_index,
                                            **add_args)[sub_index]
    pert_softmax = F.softmax(pert_vec, dim=-1)

    return 1 - torch.exp(
        -F.kl_div(out_softmax.log(), pert_softmax, None, None, 'sum')).item()
