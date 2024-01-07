import torch

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import CustomMotif


class GridMotif(CustomMotif):
    r"""Generates the grid-structured motif from the
    `"GNNExplainer: Generating Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`__ paper.
    """
    def __init__(self) -> None:
        edge_indices = [
            [0, 1],
            [0, 3],
            [1, 4],
            [3, 4],
            [1, 2],
            [2, 5],
            [4, 5],
            [3, 6],
            [6, 7],
            [4, 7],
            [5, 8],
            [7, 8],
            [1, 0],
            [3, 0],
            [4, 1],
            [4, 3],
            [2, 1],
            [5, 2],
            [5, 4],
            [6, 3],
            [7, 6],
            [7, 4],
            [8, 5],
            [8, 7],
        ]
        structure = Data(
            num_nodes=9,
            edge_index=torch.tensor(edge_indices).t().contiguous(),
            y=torch.tensor([0, 1, 0, 1, 2, 1, 0, 1, 0]),
        )
        super().__init__(structure)
