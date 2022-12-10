import torch

from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.generators import ERGraph, Motif


def test():
    # check if initializes correctly
    num_nodes, edge_prob, num_motifs = 30, 0.1, 10

    motif = Motif(structure="house")
    er_graph_gen = ERGraph(motif=motif, num_nodes=num_nodes,
                           edge_prob=edge_prob, num_motifs=num_motifs)
    dataset = ExplainerDataset(er_graph_gen)
    data = dataset[0]

    assert (data.x.size() == torch.Size(
        [num_nodes + motif.num_nodes * num_motifs, 10]))
    assert (data.y.size() == torch.Size(
        [num_nodes + motif.num_nodes * num_motifs]))
    assert (data.expl_mask.sum().item() == (num_nodes // motif.num_nodes))
    assert (data.edge_label.sum().item() == num_motifs *
            motif.edge_index.shape[1])
