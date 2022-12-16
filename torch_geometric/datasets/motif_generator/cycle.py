import torch

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import CustomMotif


class CycleMotif(CustomMotif):
    r"""Generates the cycle motif from the `"GNNExplainer:
    Generating Explanations for Graph Neural Networks"
    <https://arxiv.org/pdf/1903.03894.pdf>`_ paper, containing n nodes and n
    undirected edges.
    
    Args:
        n (int): Number of nodes (or edges) in the cycle.
    """
    def __init__(self, n: int = 6):
        # construct edge_index based on n
        structure = Data(
            num_nodes=n,
            edge_index=torch.Tensor([
                [x for x in range(n)], 
                [y for y in range(1, n)] + [0]
            ]).type(torch.int32),
        )
        super().__init__(structure)
