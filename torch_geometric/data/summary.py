from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from torch_geometric.data import Dataset


@dataclass
class Stats:
    mean: float
    std: float
    min: float
    quantile25: float
    median: float
    quantile75: float
    max: float

    @classmethod
    def from_data(cls, data: Union[List[int], List[float], torch.Tensor]):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        data = data.to(torch.float)

        return cls(
            mean=data.mean().item(),
            std=data.std().item(),
            min=data.min().item(),
            quantile25=data.quantile(0.25).item(),
            median=data.median().item(),
            quantile75=data.quantile(0.75).item(),
            max=data.max().item(),
        )


@dataclass(repr=False)
class Summary:
    name: str
    num_graphs: int
    num_nodes: Stats
    num_edges: Stats
    num_nodes_per_type: List[Tuple[str, Stats]]
    num_edges_per_type: List[Tuple[str, Stats]]

    @classmethod
    def from_dataset(cls, dataset: Dataset,
                     progress_bar: Optional[bool] = None,
                     per_type_breakdown: Optional[bool] = True):
        r"""Creates a summary of a :class:`~torch_geometric.data.Dataset`
        object.

        Args:
            dataset (Dataset): The dataset.
            progress_bar (bool, optional). If set to :obj:`True`, will show a
                progress bar during stats computation. If set to :obj:`None`,
                will automatically decide whether to show a progress bar based
                on dataset size. (default: :obj:`None`)
            per_type_breakdown (bool, optional). If set to :obj:`True`, it will
                try to separate nodes and edges and calculate stats for each
                type - can be usefull for i.e Hetero Dataset. If set to
                :obj:`False`, only the general stats for nodes and columns will
                be calculated. (default: :obj:`True`)
        """
        name = dataset.__class__.__name__

        if progress_bar is None:
            progress_bar = len(dataset) >= 10000

        node_types = []
        edge_types = []
        if hasattr(dataset, 'node_types'):
            node_types = dataset.node_types
        if hasattr(dataset, 'edge_types'):
            edge_types = dataset.edge_types

        if progress_bar:
            dataset = tqdm(dataset)

        num_nodes_list, num_edges_list = [], []
        for data in dataset:
            num_nodes_list.append(data.num_nodes)
            num_edges_list.append(data.num_edges)

        nodes_per_type = []
        edges_per_type = []
        if per_type_breakdown:
            for node_t in node_types:
                node_list = []
                for data in dataset:
                    node_list.append(data.get_node_store(node_t).num_nodes)
                nodes_per_type.append([node_t, Stats.from_data(node_list)])

            for edge_t in edge_types:
                edge_list = []
                for data in dataset:
                    edge_list.append(data.get_edge_store(*edge_t).num_edges)
                edges_per_type.append([edge_t, Stats.from_data(edge_list)])

        return cls(name=name, num_graphs=len(dataset),
                   num_nodes=Stats.from_data(num_nodes_list),
                   num_edges=Stats.from_data(num_edges_list),
                   num_nodes_per_type=nodes_per_type,
                   num_edges_per_type=edges_per_type)

    def __repr__(self) -> str:
        from tabulate import tabulate

        prefix = f'{self.name} (#graphs={self.num_graphs}):\n'

        content = [['', '#nodes', '#edges']]
        stats = [self.num_nodes, self.num_edges]
        for field in Stats.__dataclass_fields__:
            row = [field] + [f'{getattr(s, field):.1f}' for s in stats]
            content.append(row)
        body = tabulate(content, headers='firstrow', tablefmt='psql')

        if self.num_nodes_per_type:
            content = [['']]
            content[0] += [
                f'#{node_type}' for node_type, _ in self.num_nodes_per_type
            ]

            for field in Stats.__dataclass_fields__:
                row = [field] + [
                    f'{getattr(s, field):.1f}'
                    for _, s in self.num_nodes_per_type
                ]
                content.append(row)
            body += "\nNumber of nodes per node type:\n" + \
                tabulate(content, headers='firstrow', tablefmt='psql')

        if self.num_edges_per_type:
            content = [['']]
            content[0] += [
                f'#{edge_type}' for edge_type, _ in self.num_edges_per_type
            ]
            for field in Stats.__dataclass_fields__:
                row = [field] + [
                    f'{getattr(s, field):.1f}'
                    for _, s in self.num_edges_per_type
                ]
                content.append(row)
            body += "\nNumber of edges per edge type:\n" +\
                tabulate(content, headers='firstrow', tablefmt='psql')

        return prefix + body
