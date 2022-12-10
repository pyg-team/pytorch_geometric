import networkx as nx
from sklearn import metrics

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.utils import to_networkx


def get_scores(explanation, groundtruth):
    """Compute accuracy, recall, precision, f1 score, and auc of a graph.
    Args:
        explanation (Data obj): explanation graph
        groundtruth (Data obj): ground truth graph
    """
    explanation_sub = explanation.get_explanation_subgraph()
    explanation_comp = explanation.get_complement_subgraph()
    G_expl_sub = to_networkx(explanation_sub)
    G_expl_comp = to_networkx(explanation_comp)
    G_true = to_networkx(groundtruth)
    g_int = nx.intersection(G_expl_sub, G_true)
    g_int.remove_nodes_from(list(nx.isolates(g_int)))

    n_tp = g_int.number_of_edges()
    n_tn = len(G_expl_comp.edges() - g_int.edges())
    n_fp = len(G_expl_sub.edges() - g_int.edges())
    n_fn = len(G_true.edges() - g_int.edges())

    if n_tp == 0:
        precision, recall, accuracy = 0, 0, 0
        f1_score = 0
        auc = 0
    else:
        precision = n_tp / (n_tp + n_fp)
        accuracy = (n_tp + n_tn) / (n_tp + n_fp + n_tn + n_fn)
        recall = n_tp / (n_tp + n_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        fpr, tpr, thresholds = metrics.roc_curve(G_true.edges(),
                                                 G_expl_sub.edges(),
                                                 pos_label=1)
        auc = metrics.auc(fpr, tpr)

    return accuracy, recall, precision, f1_score, auc


def groundtruth_eval(explanation: Explanation, groundtruth: Data):
    """_summary_accuracy_scores: Compute accuracy scores when
    groundtruth is available"""
    accuracy, recall, precision, f1_score, auc = get_scores(
        explanation, groundtruth)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "auc": auc
    }
