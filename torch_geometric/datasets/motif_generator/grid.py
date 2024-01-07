import torch

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import CustomMotif


class GridMotif(CustomMotif):
    r"""Generates the grid-structured motif from the `"GNNExplainer:
    Generating Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`__ paper.
    """
    def __init__(self):
        structure = Data(
            num_nodes=9,
            edge_index=torch.tensor([
                [0, 0, 1, 3, 1, 2, 4, 3, 6, 4, 5, 7, 1, 3, 4, 4, 2, 5, 5, 6, 7, 7, 8, 8],
                [1, 3, 4, 4, 2, 5, 5, 6, 7, 7, 8, 8, 0, 0, 1, 3, 1, 2, 4, 3, 6, 4, 5, 7],
            ]),
            y=torch.tensor([0, 1, 0, 1, 2, 1, 0, 1, 0]),
        )
        super().__init__(structure)
        

