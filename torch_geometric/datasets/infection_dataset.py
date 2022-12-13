from typing import Any, Callable, Dict, List, Optional, Union

import torch
from networkx import all_shortest_paths, shortest_path_length

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.explain import Explanation
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph, sort_edge_index, to_networkx


class InfectionDataset(InMemoryDataset):
    r"""
    Generates a synthetic infection dataset for evaluating explainabilty
    algorithms, as described in the `"Explainability Techniques for Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.13686>`__ paper.
    The :class:`~torch_geometric.datasets.InfectionDataset` creates synthetic
    graphs coming from a
    :class:`~torch_geometric.datasets.graph_generator.GraphGenerator` with
    :obj:`num_infected` randomly assigned infected nodes. The dataset then
    creates a node classification task of predicting length of the shortest
    path from infected nodes with ground-truth node-level mask and explanation
    paths.

    For example, to generate a random Erdos-Renyi (ER) infection graph
    with 500 nodes and 0.004 probability of edge, write:

    .. code-block:: python

        from torch_geometric.datasets import InfectionDataset
        from torch_geometric.datasets.graph_generator import ERGraph

        dataset = InfectionDataset(
            graph_generator=ERGraph(num_nodes=500, edge_prob=0.004),
        )

    Args:
        graph_generator (GraphGenerator or str): The graph generator to be
            used, *e.g.*,
            :class:`torch.geometric.datasets.graph_generator.BAGraph`
            (or any string that automatically resolves to it).
        num_graphs (int, optional): The number of graphs to generate.
            (default: :obj:`1`)
        graph_generator_kwargs (Dict[str, Any], optional): Arguments passed to
            the respective graph generator module in case it gets automatically
            resolved. (default: :obj:`None`)
        seeds (List[int], optional): List of seeds to generate the infection
            graphs. (default: :obj:`None`)
        max_path_length (int, List[int]): The maximum shortest path length that
            a node will not be infected. (default: :obj:`5`)
        num_infected (List[int], optional): Number of randomly selected
            infected nodes in the graph. (default: :obj:`50`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        graph_generator: Union[GraphGenerator, str],
        num_graphs: int = 1,
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
        max_path_length: Union[int, List[int]] = 5,
        num_infected: Union[int, List[int]] = 50,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform)

        if seeds is not None:
            assert len(seeds) == num_graphs

        if isinstance(max_path_length, int):
            max_path_length = [max_path_length] * num_graphs
        assert len(max_path_length) == num_graphs
        assert all(max_path_length) >= 0

        if isinstance(num_infected, int):
            num_infected = [num_infected] * num_graphs
        assert len(num_infected) == num_graphs
        assert all(num_infected) >= 0

        self.graph_generator = GraphGenerator.resolve(
            graph_generator,
            **(graph_generator_kwargs or {}),
        )
        self.seeds = seeds
        self.max_path_length = max_path_length
        self.num_infected = num_infected

        data_list = [
            self.get_graph(
                path_length=self.max_path_length[i],
                num_infected=self.num_infected[i],
                seed=self.seeds[i] if self.seeds is not None else None)
            for i in range(num_graphs)
        ]
        self.data, self.slices = self.collate(data_list)

    def get_graph(self, path_length: int, num_infected: int,
                  seed: Optional[int] = None) -> Explanation:
        if seed is not None:
            seed_everything(seed)
        data = self.graph_generator()
        num_nodes, edge_index = data.num_nodes, data.edge_index
        rand_perm = torch.randperm(num_nodes)

        x = torch.zeros((num_nodes, 2), dtype=torch.float32)
        x[rand_perm[:num_infected], 1] = 1  # Infected
        x[rand_perm[num_infected:], 0] = 1  # Healthy
        y = torch.full((num_nodes, ), path_length, dtype=torch.long)
        node_mask = torch.zeros(num_nodes, dtype=torch.long)
        explain_path_edge_index = []
        # Original infected nodes with label 0
        y[x[:, 1].nonzero(as_tuple=True)[0]] = 0
        G = to_networkx(Data(edge_index=edge_index))
        for node in range(num_nodes):
            for distance in range(1, path_length):
                if x[node, 1] == 1 or G.in_degree(node) <= 0:
                    break
                # Get the subgraph with nodes that reach to node with k hops
                sub_nodes, _, _, _ = k_hop_subgraph(node, distance, edge_index)
                # All nodes in the subgraph are healthy
                if sum(x[sub_nodes, 1]) == 0:
                    continue
                infected_indices = x[sub_nodes, 1].nonzero(as_tuple=True)[0]
                infected_nodes = sub_nodes[infected_indices]
                sp_length = [(idx.item(),
                              shortest_path_length(G, idx.item(), node))
                             for idx in infected_nodes]
                # only retain the shortest path with distance length
                sp_length = [(idx, length) for idx, length in sp_length
                             if length == distance]
                if len(sp_length) == 0:
                    continue
                elif len(sp_length) == 1:
                    # Add explanation for node with one unique path
                    src_node = sp_length[0][0]
                    sp_generator = all_shortest_paths(G, src_node, node)
                    sp = next(sp_generator)
                    if next(sp_generator, None) is None:
                        sp_edge_index = [(sp[i], sp[i + 1])
                                         for i in range(len(sp) - 1)]
                        sp_edge_index = sort_edge_index(
                            torch.tensor(sp_edge_index, dtype=torch.long).T)
                        explain_path_edge_index.append(sp_edge_index)
                        node_mask[node] = True
                y[node] = distance
                # Prevent assigning other distance as label to this node
                break

        return Explanation(
            x=x,
            y=y,
            node_mask=node_mask,
            edge_index=edge_index,
            explain_path_edge_index=explain_path_edge_index,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'graph_generator={self.graph_generator}')
