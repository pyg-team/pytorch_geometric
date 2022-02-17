import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz


class CitationFull(InMemoryDataset):
    r"""The full citation network datasets from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
    Nodes represent documents and edges represent citation links.
    Datasets include `citeseer`, `cora`, `cora_ml`, `dblp`, `pubmed`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`, :obj:`"Cora_ML"`
            :obj:`"CiteSeer"`, :obj:`"DBLP"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Cora
              - 19,793
              - 126,842
              - 8,710
              - 70
            * - Cora_ML
              - 2,995
              - 16,316
              - 2,879
              - 7
            * - CiteSeer
              - 4,230
              - 10,674
              - 602
              - 6
            * - DBLP
              - 17,716
              - 105,734
              - 1,639
              - 4
            * - PubMed
              - 19,717
              - 88,648
              - 500
              - 3
    """

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'


class CoraFull(CitationFull):
    r"""Alias for :class:`torch_geometric.datasets.CitationFull` with
    :obj:`name="cora"`.

    Stats:
        .. list-table::
            :widths: 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - #classes
            * - 19,793
              - 126,842
              - 8,710
              - 70
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, 'cora', transform, pre_transform)

    def download(self):
        super().download()

    def process(self):
        super().process()
