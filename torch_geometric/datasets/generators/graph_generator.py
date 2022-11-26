from typing import Callable

import torch


class GraphGenerator:
    def __init__(self, motif: Callable, num_nodes: int = 300):
        self.num_nodes = num_nodes
        self.motif = motif
        self.edge_index = None
        self.edge_label = None
        self.expl_mask = None
        self.node_label = None
        self.x = None

    # TODO: FeatureGenerator (feature distribution...)
    def generate_feature(self, num_features: int = 10):
        self.x = torch.ones((self.num_nodes, num_features), dtype=torch.float)

    def attach_motif(self, num_motifs=80,
                     connection_distribution: str = 'random'):
        if connection_distribution == 'random':
            connecting_nodes = torch.randperm(self.num_nodes)[:num_motifs]
        else:
            step = self.num_nodes // num_motifs
            connecting_nodes = torch.arange(0, self.num_nodes, step)

        edge_indices = [self.edge_index]
        edge_labels = [torch.zeros(self.edge_index.size(1), dtype=torch.int64)]
        node_labels = [torch.zeros(self.num_nodes, dtype=torch.int64)]

        for i in range(num_motifs):
            edge_indices.append(self.motif.edge_index + self.num_nodes)
            edge_indices.append(
                torch.tensor([[int(connecting_nodes[i]), self.num_nodes],
                              [self.num_nodes,
                               int(connecting_nodes[i])]]))

            edge_labels.append(
                torch.ones(self.motif.edge_index.size(1), dtype=torch.long))
            edge_labels.append(torch.zeros(2, dtype=torch.long))
            node_labels.append(self.motif.label)
            self.num_nodes += self.motif.num_nodes

        self.expl_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.expl_mask[torch.arange(self.motif.num_nodes * num_motifs,
                                    self.num_nodes,
                                    self.motif.num_nodes)] = True

        self.edge_index = torch.cat(edge_indices, dim=1)
        self.edge_label = torch.cat(edge_labels, dim=0)
        self.node_label = torch.cat(node_labels, dim=0)
