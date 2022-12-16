from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import MotifGenerator
from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif
from torch_geometric.explain import Explanation


class BA2MotifDataset(InMemoryDataset):
    r"""Generates a synthetic graph classification dataset for evaluating 
    explainabilty algorithms, as described in the `"Parameterized Explainer 
    for Graph Neural Network" <https://arxiv.org/abs/2011.04573>` paper.
    The :class:`torch_geometric.datasets.BA2MotifDataset` creates
    :obj:`num_graphs`  Barabasi-Albert (BA) graphs coming from a
    :class:`torch_geometric.datasets.graph_generator.BAGraph`, and
    attaches the house motif :class:`~torch_geometric.datasets.motif_generator.HouseMotif`
    to half the :obj:`num_graphs` and the 5-node cycle motif 
    :class:`torch_geometric.datasets.motif_generator.CycleMotif` to the other half.
    The graphs with house motif are assigned graph label `0` while the one with 
    the cycle motif are assigned graph label `1`. 
    Ground-truth node-level and edge-level explainabilty masks are given based
    on whether nodes and edges are part of a certain motif or not.

    For example, to generate `100` BA graphs with num_nodes as 300 and num_edges as 5, 
    in which we want 50 to have house motifs and the other 50 to have cycle motifs,

    .. code-block:: python

        from torch_geometric.datasets import BA2MotifDataset

        dataset = BA2MotifDataset(
            num_nodes=300,
            num_edges=5,
            num_graphs=100
        )

    Args:
        num_nodes (int): The number of nodes.
            (default: :obj:`300`)
        num_edges (int): The number of edges from a new node to existing nodes.    
            (default: :obj:`5`)
        num_graphs (int, optional): The number of graphs to generate.
            (default: :obj:`100`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)        
    """
    def __init__(
        self,
        num_nodes: int = 300,
        num_edges: int = 5,
        num_graphs: int = 100,
        transform: Optional[Callable] = None,        
    ):
        super().__init__(root=None, transform=transform)
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.graph_generator = BAGraph(num_nodes=self.num_nodes, num_edges=self.num_edges)

        # initialize motif generators
        house_motif_generator = HouseMotif()
        cycle_motif_generator = CycleMotif(n = 5)

        self.num_motifs = 1

        # TODO (matthias) support on-the-fly graph generation.
        data_list = (
            [self.get_graph(house_motif_generator, label=0) for _ in range(int(num_graphs / 2))] +
            [self.get_graph(cycle_motif_generator, label=1) for _ in range(int(num_graphs / 2))]
        )
        self.data, self.slices = self.collate(data_list)  

    def get_graph(self, motif_generator: MotifGenerator, label: int) -> Explanation:
        data = self.graph_generator()

        edge_indices = [data.edge_index]
        num_nodes = data.num_nodes
        node_masks = [torch.zeros(data.num_nodes)]
        edge_masks = [torch.zeros(data.num_edges)]

        connecting_nodes = torch.randperm(num_nodes)[:self.num_motifs]
        for i in connecting_nodes.tolist():
            motif = motif_generator()

            # Add motif to the graph.
            edge_indices.append(motif.edge_index + num_nodes)
            node_masks.append(torch.ones(motif.num_nodes))
            edge_masks.append(torch.ones(motif.num_edges))

            # Add random motif connection to the graph.
            j = int(torch.randint(0, motif.num_nodes, (1, ))) + num_nodes
            edge_indices.append(torch.tensor([[i, j], [j, i]]))
            edge_masks.append(torch.zeros(2))

            num_nodes += motif.num_nodes

        return Explanation(
            edge_index=torch.cat(edge_indices, dim=1),
            y=torch.Tensor([label]).type(torch.long),
            edge_mask=torch.cat(edge_masks, dim=0),
            node_mask=torch.cat(node_masks, dim=0),
        )
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({len(self)}, '
            f'{self.graph_generator}, '
            f'num_graphs={self.num_graphs} '
            f'num_house_graphs={int(self.num_graphs / 2)} '
            f'num_cycle_graphs={int(self.num_graphs / 2)})'
        )
