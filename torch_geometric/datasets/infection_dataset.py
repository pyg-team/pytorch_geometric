from typing import Any, Callable, Dict, List, Optional, Union

import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.explain import Explanation
from torch_geometric.utils import k_hop_subgraph


class InfectionDataset(InMemoryDataset):
    r"""Generates a synthetic infection dataset for evaluating explainabilty
    algorithms, as described in the `"Explainability Techniques for Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.13686>`__ paper.
    The :class:`~torch_geometric.datasets.InfectionDataset` creates synthetic
    graphs coming from a
    :class:`~torch_geometric.datasets.graph_generator.GraphGenerator` with
    :obj:`num_infected` randomly assigned infected nodes.
    The dataset describes a node classification task of predicting the length
    of the shortest path to infected nodes, with corresponding ground-truth
    edge-level masks.

    For example, to generate a random Erdos-Renyi (ER) infection graph
    with :obj:`500` nodes and :obj:`0.004` edge probability, write:

    .. code-block:: python

        from torch_geometric.datasets import InfectionDataset
        from torch_geometric.datasets.graph_generator import ERGraph

        dataset = InfectionDataset(
            graph_generator=ERGraph(num_nodes=500, edge_prob=0.004),
            num_infected_nodes=50,
            max_path_length=3,
        )

    Args:
        graph_generator (GraphGenerator or str): The graph generator to be
            used, *e.g.*,
            :class:`torch.geometric.datasets.graph_generator.BAGraph`
            (or any string that automatically resolves to it).
        num_infected_nodes (int or List[int]): The number of randomly
            selected infected nodes in the graph.
            If given as a list, will select a different number of infected
            nodes for different graphs.
        max_path_length (int, List[int]): The maximum shortest path length to
            determine whether a node will be infected.
            If given as a list, will apply different shortest path lengths for
            different graphs. (default: :obj:`5`)
        num_graphs (int, optional): The number of graphs to generate.
            The number of graphs will be automatically determined by
            :obj:`len(num_infected_nodes)` or :obj:`len(max_path_length)` in
            case either of them is given as a list, and should only be set in
            case one wants to create multiple graphs while
            :obj:`num_infected_nodes` and :obj:`max_path_length` are given as
            an integer. (default: :obj:`None`)
        graph_generator_kwargs (Dict[str, Any], optional): Arguments passed to
            the respective graph generator module in case it gets automatically
            resolved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        graph_generator: Union[GraphGenerator, str],
        num_infected_nodes: Union[int, List[int]],
        max_path_length: Union[int, List[int]],
        num_graphs: Optional[int] = None,
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform)

        assert isinstance(num_infected_nodes, (int, list))
        assert isinstance(max_path_length, (int, list))

        if (num_graphs is None and isinstance(num_infected_nodes, int)
                and isinstance(max_path_length, int)):
            num_graphs = 1

        if num_graphs is None and isinstance(num_infected_nodes, list):
            num_graphs = len(num_infected_nodes)

        if num_graphs is None and isinstance(max_path_length, list):
            num_graphs = len(max_path_length)

        assert num_graphs is not None

        self.graph_generator = GraphGenerator.resolve(
            graph_generator,
            **(graph_generator_kwargs or {}),
        )
        self.num_infected_nodes = num_infected_nodes
        self.max_path_length = max_path_length
        self.num_graphs = num_graphs

        if isinstance(num_infected_nodes, int):
            num_infected_nodes = [num_infected_nodes] * num_graphs

        if isinstance(max_path_length, int):
            max_path_length = [max_path_length] * num_graphs

        if len(num_infected_nodes) != num_graphs:
            raise ValueError(f"The length of 'num_infected_nodes' "
                             f"(got {len(num_infected_nodes)} does not match "
                             f"with the number of graphs (got {num_graphs})")

        if len(max_path_length) != num_graphs:
            raise ValueError(f"The length of 'max_path_length' "
                             f"(got {len(max_path_length)} does not match "
                             f"with the number of graphs (got {num_graphs})")

        if any(num_infected_nodes) <= 0:
            raise ValueError(f"'num_infected_nodes' needs to be positive "
                             f"(got {min(num_infected_nodes)})")

        if any(max_path_length) <= 0:
            raise ValueError(f"'max_path_length' needs to be positive "
                             f"(got {min(max_path_length)})")

        data_list: List[Explanation] = []
        for N, L in zip(num_infected_nodes, max_path_length):
            data_list.append(self.get_graph(N, L))

        self.data, self.slices = self.collate(data_list)

    def get_graph(self, num_infected_nodes: int,
                  max_path_length: int) -> Explanation:
        data = self.graph_generator()

        assert data.num_nodes is not None
        perm = torch.randperm(data.num_nodes)
        x = torch.zeros((data.num_nodes, 2))
        x[perm[:num_infected_nodes], 1] = 1  # Infected
        x[perm[num_infected_nodes:], 0] = 1  # Healthy

        y = torch.empty(data.num_nodes, dtype=torch.long)
        y.fill_(max_path_length + 1)
        y[perm[:num_infected_nodes]] = 0  # Infected nodes have label `0`.

        assert data.edge_index is not None
        edge_mask = torch.zeros(data.num_edges, dtype=torch.bool)
        for num_hops in range(1, max_path_length + 1):
            sub_node_index, _, _, sub_edge_mask = k_hop_subgraph(
                perm[:num_infected_nodes], num_hops, data.edge_index,
                num_nodes=data.num_nodes, flow='target_to_source',
                directed=True)

            value = torch.full_like(sub_node_index, fill_value=num_hops)
            y[sub_node_index] = torch.min(y[sub_node_index], value)
            edge_mask |= sub_edge_mask

        return Explanation(
            x=x,
            edge_index=data.edge_index,
            y=y,
            edge_mask=edge_mask.to(torch.float),
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'graph_generator={self.graph_generator}, '
                f'num_infected_nodes={self.num_infected_nodes}, '
                f'max_path_length={self.max_path_length})')
