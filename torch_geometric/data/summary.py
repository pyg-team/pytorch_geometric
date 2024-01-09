from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm
from typing_extensions import Self

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.typing import EdgeType, NodeType


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
    def from_data(
        cls,
        data: Union[List[int], List[float], torch.Tensor],
    ) -> Self:
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
    num_nodes_per_type: Optional[Dict[NodeType, Stats]] = None
    num_edges_per_type: Optional[Dict[EdgeType, Stats]] = None

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        progress_bar: Optional[bool] = None,
        per_type: bool = True,
    ) -> Self:
        r"""Creates a summary of a :class:`~torch_geometric.data.Dataset`
        object.

        Args:
            dataset (Dataset): The dataset.
            progress_bar (bool, optional): If set to :obj:`True`, will show a
                progress bar during stats computation. If set to :obj:`None`,
                will automatically decide whether to show a progress bar based
                on dataset size. (default: :obj:`None`)
            per_type (bool, optional): If set to :obj:`True`, will separate
                statistics per node and edge type (only applicable in
                heterogeneous graph datasets). (default: :obj:`True`)
        """
        name = dataset.__class__.__name__

        if progress_bar is None:
            progress_bar = len(dataset) >= 10000

        if progress_bar:
            dataset = tqdm(dataset)

        num_nodes, num_edges = [], []
        _num_nodes_per_type = defaultdict(list)
        _num_edges_per_type = defaultdict(list)

        for data in dataset:
            assert data.num_nodes is not None
            num_nodes.append(data.num_nodes)
            num_edges.append(data.num_edges)

            if per_type and isinstance(data, HeteroData):
                for node_type in data.node_types:
                    _num_nodes_per_type[node_type].append(
                        data[node_type].num_nodes)
                for edge_type in data.edge_types:
                    _num_edges_per_type[edge_type].append(
                        data[edge_type].num_edges)

        num_nodes_per_type = None
        if len(_num_nodes_per_type) > 0:
            num_nodes_per_type = {
                node_type: Stats.from_data(num_nodes_list)
                for node_type, num_nodes_list in _num_nodes_per_type.items()
            }

        num_edges_per_type = None
        if len(_num_edges_per_type) > 0:
            num_edges_per_type = {
                edge_type: Stats.from_data(num_edges_list)
                for edge_type, num_edges_list in _num_edges_per_type.items()
            }

        return cls(
            name=name,
            num_graphs=len(dataset),
            num_nodes=Stats.from_data(num_nodes),
            num_edges=Stats.from_data(num_edges),
            num_nodes_per_type=num_nodes_per_type,
            num_edges_per_type=num_edges_per_type,
        )

    def __repr__(self) -> str:
        from tabulate import tabulate

        body = f'{self.name} (#graphs={self.num_graphs}):\n'

        content = [['', '#nodes', '#edges']]
        stats = [self.num_nodes, self.num_edges]
        for field in Stats.__dataclass_fields__:
            row = [field] + [f'{getattr(s, field):.1f}' for s in stats]
            content.append(row)
        body += tabulate(content, headers='firstrow', tablefmt='psql')

        if self.num_nodes_per_type is not None:
            content = [['']]
            content[0] += list(self.num_nodes_per_type.keys())

            for field in Stats.__dataclass_fields__:
                row = [field] + [
                    f'{getattr(s, field):.1f}'
                    for s in self.num_nodes_per_type.values()
                ]
                content.append(row)
            body += "\nNumber of nodes per node type:\n"
            body += tabulate(content, headers='firstrow', tablefmt='psql')

        if self.num_edges_per_type is not None:
            content = [['']]
            content[0] += [
                f"({', '.join(edge_type)})"
                for edge_type in self.num_edges_per_type.keys()
            ]

            for field in Stats.__dataclass_fields__:
                row = [field] + [
                    f'{getattr(s, field):.1f}'
                    for s in self.num_edges_per_type.values()
                ]
                content.append(row)
            body += "\nNumber of edges per edge type:\n"
            body += tabulate(content, headers='firstrow', tablefmt='psql')

        return body
