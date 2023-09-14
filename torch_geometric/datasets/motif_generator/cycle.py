import torch

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import CustomMotif


class CycleMotif(CustomMotif):
    r"""Generates the cycle motif from the `"GNNExplainer:
    Generating Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`__ paper.

    Args:
        num_nodes (int): The number of nodes in the cycle.
    """
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

        row = torch.arange(num_nodes).view(-1, 1).repeat(1, 2).view(-1)
        col1 = torch.arange(-1, num_nodes - 1) % num_nodes
        col2 = torch.arange(1, num_nodes + 1) % num_nodes
        col = torch.stack([col1, col2], dim=1).sort(dim=-1)[0].view(-1)

        structure = Data(
            num_nodes=num_nodes,
            edge_index=torch.stack([row, col], dim=0),
        )

        super().__init__(structure)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_nodes})'
