from dataclasses import dataclass
from typing import List, Optional, Union

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

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        progress_bar: Optional[bool] = None,
    ):
        r"""Creates a summary of a :class:`~torch_geometric.data.Dataset`
        object.

        Args:
            dataset (Dataset): The dataset.
            progress_bar (bool, optional). If set to :obj:`True`, will show a
                progress bar during stats computation. If set to :obj:`None`,
                will automatically decide whether to show a progress bar based
                on dataset size. (default: :obj:`None`)
        """
        if progress_bar is None:
            progress_bar = len(dataset) >= 10000

        if progress_bar:
            dataset = tqdm(dataset)

        num_nodes_list, num_edges_list = [], []
        for data in dataset:
            num_nodes_list.append(data.num_nodes)
            num_edges_list.append(data.num_edges)

        return cls(
            name=dataset.__class__.__name__,
            num_graphs=len(dataset),
            num_nodes=Stats.from_data(num_nodes_list),
            num_edges=Stats.from_data(num_edges_list),
        )

    def __repr__(self) -> str:
        from tabulate import tabulate

        prefix = f'{self.name} (#graphs={self.num_graphs}):\n'

        content = [['', '#nodes', '#edges']]
        stats = [self.num_nodes, self.num_edges]
        for field in Stats.__dataclass_fields__:
            row = [field] + [f'{getattr(s, field):.1f}' for s in stats]
            content.append(row)
        body = tabulate(content, headers='firstrow', tablefmt='psql')

        return prefix + body
