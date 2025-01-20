from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_greyc
from torch_geometric.utils import from_networkx


class DatasetNotFoundError(Exception):
    pass


class GreycDataset(InMemoryDataset):
    r"""Class to load three GREYC Datasets as pytorch geometric dataset."""

    URL = ('https://raw.githubusercontent.com/bgauzere/greycdata/refs/'
           'heads/main/greycdata/data/')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __str__(self) -> str:
        return self.name

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def raw_file_names(self) -> List[str]:
        return []

    def download(self) -> None:
        """Load the right data according to initializer."""
        if self.name == 'alkane':
            download_url(GreycDataset.URL + "Aklane", self.raw_dir)
        elif self.name == 'acyclic':
            download_url(GreycDataset.URL + "Acyclic", self.raw_dir)
        elif self.name == 'mao':
            download_url(GreycDataset.URL + "MAO", self.raw_dir)
        else:
            raise DatasetNotFoundError(f"Dataset `{self.name}` not found")

    def process(self):
        """Read data into huge `Data` list."""
        graph_list, property_list = read_greyc(self.raw_dir, self.name)

        # Convert to PyG.

        def from_nx_to_pyg(graph, y):
            """Convert networkx graph to pytorch graph and add y."""
            pyg_graph = from_networkx(
                graph,
                group_node_attrs=['atom_symbol', 'degree', 'x', 'y', 'z'])
            pyg_graph.y = y
            return pyg_graph

        data_list = [
            from_nx_to_pyg(graph, y)
            for graph, y in zip(graph_list, property_list)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
