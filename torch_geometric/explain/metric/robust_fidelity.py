from typing import Tuple

import numpy as np

import torch

from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import ModelMode

from torch_geometric.utils import k_hop_subgraph
from scipy.sparse import coo_matrix


def robust_fidelity(
        explainer: Explainer,
        explanation: Explanation,
        alpha1=0.1,
        alpha2=0.9,
        sample_num=50,
        top_k=-1,
        k_hop=3,
        undirect=True,
        use_gt_label=True
) -> Tuple[float, float, float, float, float, float]:
    r"""Calculate the robust fidelity  metric, given an
    :class:`~torch_geometric.explain.Explainer`  and
    :class:`~torch_geometric.explain.Explanation`, as described in the
    `"Towards Robust Fidelity for Evaluating Explainability of Graph
    Neural Networks" <https://arxiv.org/abs/2310.01820>`_ paper.

    Fidelity is a metric that evaluates the contribution of the given
    explanation subgraph to the original prediction. However, due to
    the prediction function may not be trained in the distribution
    of explanation subgraphs, the fidelity might be not accurate.
    Robust Fidelity allievate the Out-of-Distbution problem in the
    fidelity metric. Similar to fidelity, this function return two
    scores  by giving only the subgraph to the model (fidelity-) or
    by removing it from the entire graph (fidelity+).

    the probability-based robust fidelity scores are given by:

    .. math::
                Fid_{\alpha_1,+} &= f(\overline{G})_y -
                \mathbb{E}f(\overline{G}-
                E_{\alpha_1}(\overline{G}^{(exp)}))_y

                Fid_{\alpha_2,-} &=  f(\overline{G})_y -
                \mathbb{E}f(\overline{G}^{(exp)}+
                E_{\alpha_2}(\overline{G}-\overline{G}^{(exp)}))_y

                Fid_{\alpha_1,\alpha_2,\Delta} &=
                Fid_{\alpha_1,+} - Fid_{\alpha_2,-}

    the accuracy-based robust fidelity scores are given by:

    .. math::
                Fid_{\alpha_1,+} &= \mathbb{1}(
                \widehat{y}_{\overline{G}} == y ) -
                \mathbb{E}( \mathbb{1}( \widehat{y}_
                {\overline{G}-E_{\alpha_1}
                (\overline{G}^{(exp)})} == y))

                Fid_{\alpha_2,-} &=  \mathbb{1}(
                \widehat{y}_{\overline{G}} == y ) -
                \mathbb{E}( \mathbb{1}( \widehat{y}_
                {\overline{G}^{(exp)}+E_{\alpha_2}(
                \overline{G}-\overline{G}^{(exp)})}==y))

                Fid_{\alpha_1,\alpha_2,\Delta} &=
                Fid_{\alpha_1,+} - Fid_{\alpha_2,-}

    this method is designed for edge-based explanation subgraphs,
    node-based explanation subgraphs should convert into
    edge-based explanation subgraphs.

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
        alpha1: the ratio of remove explanation subgraph each
                time in fid+ calculation
        alpha2: the ratio of maintain non-explanation subgraph
                each time in fid- calculation
        k_hop: the number of hop for node classification
        undirect: if the graph is undirected graph (default:
                graph task is true, node task is false)
        use_gt_label: use gt_label to calculate the fid

    """
    max_length = sample_num
    alpha2 = 1 - alpha2
    task_type = 'node' if explanation.get('index') is not None else 'graph'
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    edge_mask_np = explanation.get('edge_mask').cpu().detach().numpy()
    if top_k != -1:
        idx = np.argpartition(edge_mask_np, top_k)
        edge_mask_np = np.where(edge_mask_np > edge_mask_np[idx],
                                np.ones_like(edge_mask_np),
                                np.zeros_like(edge_mask_np))

    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.target
    y_hat = explainer.get_prediction(
        explanation.x,
        explanation.edge_index,
        **kwargs,
    )
    y_label = explainer.get_target(y_hat)  # original label

    features = explanation.x
    graphs = explanation.edge_index

    matrix_0 = graphs[0].cpu().numpy()
    matrix_1 = graphs[1].cpu().numpy()
    exp_graph_matrix = coo_matrix(
        (edge_mask_np,
         (matrix_0, matrix_1)),
        shape=(features.shape[0], features.shape[0])).tocsr()

    if task_type == 'node':
        index = explanation.index

        y = y[index].view(-1)
        y_hat = y_hat[index].view(-1)
        y_label = y_label[index].view(-1)

        subset, edge_index, mapping, edge_mask_ = \
            k_hop_subgraph(index, k_hop, graphs, relabel_nodes=False)
        edge_index_np = edge_index.cpu().detach().numpy()
        sample_matrix = coo_matrix(
            (np.ones_like(edge_index_np[0]),
             (edge_index_np[0], edge_index_np[1])),
            shape=(features.shape[0], features.shape[0])).tocsr()

        graph_matrix = sample_matrix.multiply(exp_graph_matrix)
        non_graph_matrix = sample_matrix - graph_matrix
        weights = graph_matrix[edge_index_np[0], edge_index_np[1]].A[0]
        explain = torch.tensor(weights).float().to(graphs.device)
        weights = non_graph_matrix[edge_index_np[0], edge_index_np[1]].A[0]
        non_explain = torch.tensor(weights).float().to(graphs.device)
    else:
        weights = edge_mask_np
        explain = torch.tensor(weights).float().to(graphs.device)
        non_explain = torch.tensor(1 - weights).float().to(graphs.device)

    if undirect:
        maps = {}
        explain_list = []
        non_explain_list = []
        for i, (nodeid0, nodeid1, ex) in \
                enumerate(zip(matrix_0, matrix_1, edge_mask_np)):
            max_node = max(nodeid0, nodeid1)
            min_node = min(nodeid0, nodeid1)
            if (min_node, max_node) in maps.keys():
                maps[(min_node, max_node)].append(i)
                if ex > 0.5:
                    explain_list.append((min_node, max_node))
                else:
                    non_explain_list.append((min_node, max_node))
            else:
                maps[(min_node, max_node)] = [i]

    else:
        explain_list = \
            torch.nonzero(explain).cpu().detach().numpy().tolist()
        non_explain_list = \
            torch.nonzero(non_explain).cpu().detach().numpy().tolist()

    if use_gt_label:
        label = int(y)
    else:
        label = int(y_label)

    explaine_ratio = np.ones(len(explain_list))
    explaine_ratio = \
        alpha1 * explaine_ratio.sum() * \
        (explaine_ratio / explaine_ratio.sum())
    explaine_ratio_remove = \
        np.random.binomial(1, explaine_ratio,
                           size=(max_length, explaine_ratio.shape[0]))

    non_explaine_ratio = np.ones(len(non_explain_list))
    non_explaine_ratio = \
        alpha2 * non_explaine_ratio.sum() * \
        (non_explaine_ratio / non_explaine_ratio.sum())
    non_explaine_ratio_remove = \
        np.random.binomial(1, non_explaine_ratio,
                           size=(max_length, non_explaine_ratio.shape[0]))

    def cal_fid_embedding_plus():
        list_explain = torch.zeros([max_length, explain.shape[0]])
        for i in range(max_length):
            remove_edges = explaine_ratio_remove[i]
            for idx, edge in enumerate(explain_list):
                if remove_edges[idx] == 1:
                    if undirect:
                        id_lists = maps[edge]
                        for id in id_lists:
                            list_explain[i, id] = 1.0
                    else:
                        list_explain[i, idx] = 1.0

        fid_plus_prob_list = []
        fid_plus_acc_list = []

        for i in range(max_length):
            if task_type == 'node':
                with torch.no_grad():
                    mask_pred_plus = explainer.get_masked_prediction(
                        features,
                        edge_index,
                        1. - node_mask if node_mask is not None else None,
                        1. - list_explain[i].to(features.device)
                        if edge_mask is not None else None,
                        **kwargs,
                    )
                    mask_pred_plus_label = explainer.get_target(mask_pred_plus)
                    mask_pred_plus = mask_pred_plus[index].view(-1)

                    mask_label_plus = mask_pred_plus_label[index].view(-1)

                    fid_plus = y_hat[label] - mask_pred_plus[label]
                    fid_plus_label = \
                        int(y_label == label) - int(mask_label_plus == label)

            else:
                with torch.no_grad():
                    mask_pred_plus = explainer.get_masked_prediction(
                        features,
                        graphs,
                        1. - node_mask if node_mask is not None else None,
                        1. - list_explain[i].to(features.device)
                        if edge_mask is not None else None,
                        **kwargs,
                    )

                    mask_pred_plus_label = explainer.get_target(mask_pred_plus)

                    mask_label_plus = mask_pred_plus_label

                    fid_plus = y_hat[:, label] - mask_pred_plus[:, label]
                    fid_plus_label = \
                        int(y_label == label) - int(mask_label_plus == label)

            fid_plus_prob_list.append(fid_plus)
            fid_plus_acc_list.append(fid_plus_label)
        if len(fid_plus_prob_list) < 1:
            return 0, 0
        else:
            fid_plus_mean = \
                torch.stack(fid_plus_prob_list).mean().cpu().detach().numpy()
            fid_plus_label_mean = np.stack(fid_plus_acc_list).mean()
        return fid_plus_mean, fid_plus_label_mean

    def cal_fid_embedding_minus():
        # global non_explain_indexs_combin
        list_explain = torch.zeros([max_length, non_explain.shape[0]])
        for i in range(max_length):
            remove_edges = non_explaine_ratio_remove[i]
            for idx, edge in enumerate(non_explain_list):
                if remove_edges[idx] == 1:
                    if undirect:
                        id_lists = maps[edge]  # get two edges id
                        for id in id_lists:
                            list_explain[i, id] = 1.0
                    else:
                        list_explain[i, idx] = 1.0

        fid_minus_prob_list = []
        fid_minus_acc_list = []
        # fid_minus_embedding_distance_list = []

        for i in range(max_length):
            if task_type == 'node':
                with torch.no_grad():
                    mask_pred_minus = explainer.get_masked_prediction(
                        features,
                        edge_index,
                        node_mask,
                        list_explain[i].to(features.device),
                        **kwargs, )
                    mask_pred_minus_label = \
                        explainer.get_target(mask_pred_minus)

                    mask_pred_minus = mask_pred_minus[index].view(-1)
                    mask_label_minus = mask_pred_minus_label[index].view(-1)

                    fid_minus = y_hat[label] - mask_pred_minus[label]
                    fid_minus_label = \
                        int(y_label == label) - int(mask_label_minus == label)

            else:
                with torch.no_grad():
                    mask_pred_minus = explainer.get_masked_prediction(
                        features,
                        graphs,
                        node_mask,
                        list_explain[i].to(features.device),
                        **kwargs, )
                    mask_pred_minus_label = \
                        explainer.get_target(mask_pred_minus)
                    mask_label_minus = mask_pred_minus_label

                    fid_minus = y_hat[:, label] - mask_pred_minus[:, label]
                    fid_minus_label = \
                        int(y_label == label) - int(mask_label_minus == label)

            fid_minus_prob_list.append(fid_minus)
            fid_minus_acc_list.append(fid_minus_label)

        if len(fid_minus_prob_list) < 1:
            return 1, 1
        else:
            fid_minus_mean = \
                torch.stack(fid_minus_prob_list).mean().cpu().detach().numpy()
            fid_minus_label_mean = np.stack(fid_minus_acc_list).mean()
        return fid_minus_mean, fid_minus_label_mean

    fid_plus_mean, fid_plus_label_mean = cal_fid_embedding_plus()
    fid_minus_mean, fid_minus_label_mean = cal_fid_embedding_minus()
    fid_delta = fid_plus_mean-fid_minus_mean
    fid_delta_label = fid_plus_label_mean - fid_minus_label_mean

    return \
        fid_plus_mean, fid_minus_mean, fid_delta, \
        fid_plus_label_mean, fid_minus_label_mean, fid_delta_label
