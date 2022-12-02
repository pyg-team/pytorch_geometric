from typing import Callable, Optional

import torch

from torch_geometric.data import Data


class GraphGenerator:
    def __init__(
        self,
        num_nodes: int = 300,
        motif: Optional[Callable] = None,
    ):
        self.num_nodes = num_nodes
        self.motif = motif
        self._edge_index = None
        self._edge_label = None
        self._expl_mask = None
        self._node_label = None
        self._x = None

    def generate_base_graph(self) -> Data:
        raise NotImplementedError

    @property
    def data(self):
        return Data(x=self._x, edge_index=self._edge_index, y=self._node_label,
                    expl_mask=self._expl_mask, edge_label=self._edge_label)

    # TODO: FeatureGenerator (feature distribution...)
    def generate_feature(self, num_features: int = 10):
        self._x = torch.ones((self.num_nodes, num_features), dtype=torch.float)

    def attach_motif(self, num_motifs=80):
        if not self.motif:
            return

        connecting_nodes = torch.randperm(self.num_nodes)[:num_motifs]
        edge_indices = [self._edge_index]
        edge_labels = [
            torch.zeros(self._edge_index.size(1), dtype=torch.int64)
        ]
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

        self._expl_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self._expl_mask[torch.arange(self.motif.num_nodes * num_motifs,
                                     self.num_nodes,
                                     self.motif.num_nodes)] = True

        self._edge_index = torch.cat(edge_indices, dim=1)
        self._edge_label = torch.cat(edge_labels, dim=0)
        self._node_label = torch.cat(node_labels, dim=0)
