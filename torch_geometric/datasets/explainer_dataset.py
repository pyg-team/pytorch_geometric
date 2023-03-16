from typing import Any, Callable, Dict, Optional, Union

import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.datasets.motif_generator import MotifGenerator
from torch_geometric.explain import Explanation


class ExplainerDataset(InMemoryDataset):
    r"""Generates a synthetic dataset for evaluating explainabilty algorithms,
    as described in the `"GNNExplainer: Generating Explanations for Graph
    Neural Networks" <https://arxiv.org/abs/1903.03894>`__ paper.
    The :class:`~torch_geometric.datasets.ExplainerDataset` creates synthetic
    graphs coming from a
    :class:`~torch_geometric.datasets.graph_generator.GraphGenerator`, and
    randomly attaches :obj:`num_motifs` many motifs to it coming from a
    :class:`~torch_geometric.datasets.graph_generator.MotifGenerator`.
    Ground-truth node-level and edge-level explainabilty masks are given based
    on whether nodes and edges are part of a certain motif or not.

    For example, to generate a random Barabasi-Albert (BA) graph with 300
    nodes, in which we want to randomly attach 80 :obj:`"house"` motifs, write:

    .. code-block:: python

        from torch_geometric.datasets import ExplainerDataset
        from torch_geometric.datasets.graph_generator import BAGraph

        dataset = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=300, num_edges=5),
            motif_generator='house',
            num_motifs=80,
        )

    .. note::

        For an example of using :class:`ExplainerDataset`, see
        `examples/explain/gnn_explainer_ba_shapes.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        /explain/gnn_explainer_ba_shapes.py>`_.

    Args:
        graph_generator (GraphGenerator or str): The graph generator to be
            used, *e.g.*,
            :class:`torch.geometric.datasets.graph_generator.BAGraph`
            (or any string that automatically resolves to it).
        motif_generator (MotifGenerator): The motif generator to be used,
            *e.g.*,
            :class:`torch_geometric.datasets.motif_generator.HouseMotif`
            (or any string that automatically resolves to it).
        num_motifs (int): The number of motifs to attach to the graph.
        num_graphs (int, optional): The number of graphs to generate.
            (default: :obj:`1`)
        graph_generator_kwargs (Dict[str, Any], optional): Arguments passed to
            the respective graph generator module in case it gets automatically
            resolved. (default: :obj:`None`)
        motif_generator_kwargs (Dict[str, Any], optional): Arguments passed to
            the respective motif generator module in case it gets automatically
            resolved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        graph_generator: Union[GraphGenerator, str],
        motif_generator: Union[MotifGenerator, str],
        num_motifs: int,
        num_graphs: int = 1,
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        motif_generator_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform)

        if num_motifs <= 0:
            raise ValueError(f"At least one motif needs to be attached to the "
                             f"graph (got {num_motifs})")

        self.graph_generator = GraphGenerator.resolve(
            graph_generator,
            **(graph_generator_kwargs or {}),
        )
        self.motif_generator = MotifGenerator.resolve(
            motif_generator,
            **(motif_generator_kwargs or {}),
        )
        self.num_motifs = num_motifs

        # TODO (matthias) support on-the-fly graph generation.
        data_list = [self.get_graph() for _ in range(num_graphs)]
        self.data, self.slices = self.collate(data_list)

    def get_graph(self) -> Explanation:
        data = self.graph_generator()

        edge_indices = [data.edge_index]
        num_nodes = data.num_nodes
        node_masks = [torch.zeros(data.num_nodes)]
        edge_masks = [torch.zeros(data.num_edges)]
        ys = [torch.zeros(num_nodes, dtype=torch.long)]

        connecting_nodes = torch.randperm(num_nodes)[:self.num_motifs]
        for i in connecting_nodes.tolist():
            motif = self.motif_generator()

            # Add motif to the graph.
            edge_indices.append(motif.edge_index + num_nodes)
            node_masks.append(torch.ones(motif.num_nodes))
            edge_masks.append(torch.ones(motif.num_edges))

            # Add random motif connection to the graph.
            j = int(torch.randint(0, motif.num_nodes, (1, ))) + num_nodes
            edge_indices.append(torch.tensor([[i, j], [j, i]]))
            edge_masks.append(torch.zeros(2))

            if 'y' in motif:
                ys.append(motif.y + 1 if motif.y.min() == 0 else motif.y)

            num_nodes += motif.num_nodes

        return Explanation(
            edge_index=torch.cat(edge_indices, dim=1),
            y=torch.cat(ys, dim=0) if len(ys) > 1 else None,
            edge_mask=torch.cat(edge_masks, dim=0),
            node_mask=torch.cat(node_masks, dim=0),
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'graph_generator={self.graph_generator}, '
                f'motif_generator={self.motif_generator}, '
                f'num_motifs={self.num_motifs})')
