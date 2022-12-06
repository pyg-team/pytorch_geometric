
from typing import Tuple

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from utils import perturb_node_features


def get_faithfulness(data: Data, explanation: Explanation, model: torch.nn.Module, topk: int, node_idx: int = None, **kwargs) -> Tuple[float, float]:
    r"""

    Faithfulness sore for explainability
    Based on Evaluating Explainability for Graph Neural Networks
    @misc{https://doi.org/10.48550/arxiv.2208.09339,
      doi = {10.48550/ARXIV.2208.09339},
      url = {https://arxiv.org/abs/2208.09339},
      author = {Agarwal, Chirag and Queen, Owen and Lakkaraju, Himabindu and Zitnik, Marinka},
      title = {Evaluating Explainability for Graph Neural Networks},
      code = {https://github.com/mims-harvard/GraphXAI},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution 4.0 International}

    }


    we obtain the prediction probability vector ŷ_u using the GNN, i.e. ŷ_u = f (S_u),
    and using the explanation, i.e. ŷ_u′ = f (S_u′), where we generate a masked subgraph S_u′ by only keeping the original
    values of the top-k features identified by an explanation, and get their respective predictions ŷ_u′

    Un-Faithfulness of a GNN predictor is given as
    GEF(f,S_u,S_u') = 1 - exp^-KL(ŷ_u || ŷ_u')

    Note that higher values indicate a higher degree of unfaithfulness


    Args:
        node_idx (int): node_idx for node to explain, should be the same used for explainer (it can be get from explainer)
        data (Data) : dataset for node level tasks / data for one graph for which we would like to calculate faithfulness fpr gtaph level tasks
        explanation (Explanation): Explanation output generated by an explainer
        model (torch.nn.Module): model used in explanation (ait can be get from explanation)
        topk (int): k most important feature/nodes, it does not calculate percentage and
                    the user should take into account the threshold config used in explanation
        kwargs (dict, optional): Any additional arguments to the forward method
            of model, other than x and edge_index.
    """
    node_faithfulness = node_feat_faithfulness = None
    device = data.x.device
    try:
        with torch.no_grad():
            if node_idx is None:
                org_vec = model(data.x, data.edge_index, **kwargs)
            else:
                org_vec = model(data.x, data.edge_index, **kwargs)[node_idx]

            org_softmax = F.softmax(org_vec[node_idx], dim=-1)
    except:
        raise Exception("Not able to get get original predictions from model and data")

    if getattr(explanation, 'node_mask', None) is not None:
        if topk > explanation.node_mask.shape[0]:
            raise ValueError("Topk cannot be greater than nodes size, take also in consideration threshold config in explainer")

        # getting top k important nodes according to explanation
        top_k_nodes = explanation.node_mask.topk(topk)[1]

        pert_x = data.x.clone()
        # removing not top-k
        rem_nodes = [node for node in range(data.x.shape[0]) if node not in top_k_nodes]
        pert_x[rem_nodes] = torch.zeros_like(pert_x[rem_nodes]).to(device)
        # getting predictions of explainer
        if node_idx is not None:
            pert_vec = model(pert_x, data.edge_index, **kwargs)[node_idx]
        else:
            pert_vec = model(pert_x, data.edge_index, **kwargs)
        pert_softmax = F.softmax(pert_vec, dim=-1)

        # calculation faithfulness
        node_faithfulness = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    if getattr(explanation, 'node_feat_mask', None) is not None:

        if topk > explanation.node_feat_mask.shape[1]:
            raise ValueError("Topk cannot be greater than feature dimension, take also in consideration threshold config in explainer")

        # getting the top_k features in the node attribute feature vector based on explanation
        top_k_features = explanation.node_feat_mask[node_idx].topk(topk)[1]
        pert_x = data.x.clone().to(device)

        # Perturbing the unimportant node features
        rem_features = torch.Tensor(
            [i for i in range(data.x.shape[1]) if i not in top_k_features]).long().to(device)

        pert_x[node_idx, rem_features] = perturb_node_features(x=pert_x, node_idx=node_idx, pert_feat=rem_features,
                                                               device=device)

        pert_vec = model(pert_x, data.edge_index, **kwargs)[node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)

        node_feat_faithfulness = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    return node_faithfulness, node_feat_faithfulness

