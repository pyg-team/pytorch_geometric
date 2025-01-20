import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_zip


class DatasetNotFoundError(Exception):
    pass


class GreycDataset(InMemoryDataset):
    r"""Class to load three GREYC Datasets as pytorch geometric dataset."""

    URL = ('http://localhost:3000/')

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
        path = download_url(GreycDataset.URL + self.name + ".zip", self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        """Read data into huge `Data` list."""
        dataset = torch.load(self.raw_dir + self.name + ".pth")

        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])
