import torch

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import CustomMotif


class HouseMotif(CustomMotif):
    r"""Generates the house-structured motif from the `"GNNExplainer:
    Generating Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`__ paper, containing 5 nodes and 6
    undirected edges. Nodes are labeled according to their structural role:
    the top, middle and bottom of the house."""
    def __init__(self):
        structure = Data(
            num_nodes=5,
            edge_index=torch.tensor([
                [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1],
            ]),
            y=torch.tensor([0, 0, 1, 1, 2]),
        )
        super().__init__(structure)
