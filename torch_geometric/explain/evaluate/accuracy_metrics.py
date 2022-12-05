import networkx as nx

from torch_geometric.utils import to_networkx


def get_scores(G1, G2):
    """Compute accuracy, recall, precision, f1 score, and auc of a graph.
    Args:
        G1 (networkx graph): explanation graph
        G2 (networkx graph): ground truth graph
    """
    G1, G2 = G1.to_undirected(), G2.to_undirected()
    g_int = nx.intersection(G1, G2)
    g_int.remove_nodes_from(list(nx.isolates(g_int)))

    n_tp = g_int.number_of_edges()
    n_fp = len(G1.edges() - g_int.edges())
    n_fn = len(G2.edges() - g_int.edges())

    if n_tp == 0:
        precision, recall, accuracy = 0, 0, 0
        f1_score = 0
        auc = 0
    else:
        precision = n_tp / (n_tp + n_fp)
        # accuracy = (n_tp + n_tn)/(n_tp + n_fp + n_tn + n_fn)
        recall = n_tp / (n_tp + n_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        # fpr, tpr, thresholds = metrics.roc_curve(G1.edges(), G2.edges(),
        # pos_label=1)
        # auc = metrics.auc(fpr, tpr)

    return accuracy, recall, precision, f1_score, auc


def groundtruth_eval(explanation, groundtruth):
    """_summary_accuracy_scores: Compute accuracy scores when
    groundtruth is available"""
    G_true = to_networkx(groundtruth)
    G_expl = to_networkx(explanation)
    accuracy, recall, precision, f1_score, auc = get_scores(G_expl, G_true)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "auc": auc
    }
